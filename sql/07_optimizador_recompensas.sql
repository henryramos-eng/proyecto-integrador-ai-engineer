/* Contrato de persistencia para etapa 4: optimizador de recompensas. */
SET NOCOUNT ON;
SET XACT_ABORT ON;

IF DB_NAME() <> N'CasinoPalacioReal'
    THROW 51030, 'Este script debe ejecutarse en CasinoPalacioReal.', 1;
IF SCHEMA_ID(N'ml') IS NULL OR OBJECT_ID(N'ml.ResponseModelRun',N'U') IS NULL
    THROW 51031, 'Debe completarse primero la etapa 3 de respuesta.', 1;

IF OBJECT_ID(N'ml.RewardAllocationRun',N'U') IS NULL
BEGIN
    CREATE TABLE ml.RewardAllocationRun (
        AllocationRunId bigint IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ml_RewardAllocationRun PRIMARY KEY,
        ResponseModelRunId bigint NOT NULL,
        OptimizerVersion varchar(60) NOT NULL,
        SourceFingerprint char(64) NOT NULL,
        RiskSource varchar(100) NOT NULL,
        RiskArtifactHash char(64) NOT NULL,
        BaselineArtifactHash char(64) NOT NULL,
        Budget decimal(19,6) NOT NULL,
        MaxHighFraction decimal(9,6) NOT NULL,
        MaxSegmentFraction decimal(9,6) NOT NULL,
        MinSessions90Days int NOT NULL,
        InputClients int NOT NULL,
        PositiveCandidates int NOT NULL,
        AssignedClients int NOT NULL,
        TotalSpend decimal(19,6) NOT NULL,
        TotalExpectedValue decimal(19,6) NOT NULL,
        BaselineAssignedClients int NOT NULL,
        BaselineSpend decimal(19,6) NOT NULL,
        BaselineExpectedValue decimal(19,6) NOT NULL,
        MetricsJson nvarchar(max) NOT NULL,
        Status varchar(20) NOT NULL,
        StartedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_AllocationRun_StartedAt DEFAULT SYSUTCDATETIME(),
        CompletedAt datetime2(0) NULL,
        CONSTRAINT FK_ml_AllocationRun_ResponseRun FOREIGN KEY(ResponseModelRunId)
            REFERENCES ml.ResponseModelRun(ResponseModelRunId),
        CONSTRAINT UQ_ml_AllocationRun_VersionSource
            UNIQUE(OptimizerVersion,SourceFingerprint),
        CONSTRAINT CK_ml_AllocationRun_Status
            CHECK(Status IN ('STARTED','COMPLETED','FAILED')),
        CONSTRAINT CK_ml_AllocationRun_Budget CHECK(Budget > 0 AND TotalSpend <= Budget),
        CONSTRAINT CK_ml_AllocationRun_Fractions CHECK(
            MaxHighFraction BETWEEN 0 AND 1 AND MaxSegmentFraction BETWEEN 0 AND 1
        ),
        CONSTRAINT CK_ml_AllocationRun_Counts CHECK(
            InputClients >= 0 AND PositiveCandidates >= 0 AND AssignedClients >= 0
            AND AssignedClients <= PositiveCandidates AND PositiveCandidates <= InputClients
        )
    );
END;

IF OBJECT_ID(N'ml.RewardAllocationDecision',N'U') IS NULL
BEGIN
    CREATE TABLE ml.RewardAllocationDecision (
        AllocationRunId bigint NOT NULL,
        IdCliente bigint NOT NULL,
        Segment varchar(50) NULL,
        RiskLevel varchar(10) NOT NULL,
        RiskScore decimal(19,10) NULL,
        SessionsLast90Days int NOT NULL,
        SuggestedReward varchar(10) NULL,
        SuggestedCost decimal(19,6) NOT NULL,
        AssignedReward varchar(10) NULL,
        Cost decimal(19,6) NOT NULL,
        ResponseProbability decimal(12,10) NULL,
        IncrementalValue decimal(19,6) NULL,
        ExpectedValue decimal(19,6) NULL,
        Efficiency decimal(19,10) NULL,
        CumulativeSpend decimal(19,6) NULL,
        Assigned bit NOT NULL,
        DecisionReason nvarchar(250) NOT NULL,
        BaselineAssignedReward varchar(10) NULL,
        BaselineCost decimal(19,6) NOT NULL,
        BaselineExpectedValue decimal(19,6) NULL,
        BaselineAssigned bit NOT NULL,
        CreatedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_AllocationDecision_CreatedAt DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_ml_RewardAllocationDecision PRIMARY KEY(AllocationRunId,IdCliente),
        CONSTRAINT FK_ml_AllocationDecision_Run FOREIGN KEY(AllocationRunId)
            REFERENCES ml.RewardAllocationRun(AllocationRunId),
        CONSTRAINT CK_ml_AllocationDecision_Risk CHECK(RiskLevel IN ('Bajo','Medio','Alto')),
        CONSTRAINT CK_ml_AllocationDecision_SuggestedReward CHECK(
            SuggestedReward IS NULL OR SuggestedReward IN ('baja','media','alta')
        ),
        CONSTRAINT CK_ml_AllocationDecision_AssignedReward CHECK(
            AssignedReward IS NULL OR AssignedReward IN ('baja','media','alta')
        ),
        CONSTRAINT CK_ml_AllocationDecision_BaselineReward CHECK(
            BaselineAssignedReward IS NULL OR BaselineAssignedReward IN ('baja','media','alta')
        ),
        CONSTRAINT CK_ml_AllocationDecision_Probability CHECK(
            ResponseProbability IS NULL OR ResponseProbability BETWEEN 0 AND 1
        ),
        CONSTRAINT CK_ml_AllocationDecision_Costs CHECK(
            SuggestedCost >= 0 AND Cost >= 0 AND BaselineCost >= 0
        ),
        CONSTRAINT CK_ml_AllocationDecision_Assignment CHECK(
            (Assigned=1 AND AssignedReward IS NOT NULL AND Cost > 0 AND ExpectedValue > 0)
            OR (Assigned=0 AND AssignedReward IS NULL AND Cost=0)
        )
    );
    CREATE INDEX IX_ml_AllocationDecision_RunAssignedReward
        ON ml.RewardAllocationDecision(AllocationRunId,Assigned,RiskLevel,AssignedReward);
    CREATE INDEX IX_ml_AllocationDecision_RunReason
        ON ml.RewardAllocationDecision(AllocationRunId,DecisionReason);
END;

EXEC(N'
CREATE OR ALTER VIEW ml.vw_RewardAllocationCurrent AS
SELECT d.*
FROM ml.RewardAllocationDecision d
JOIN (
    SELECT TOP(1) AllocationRunId
    FROM ml.RewardAllocationRun WHERE Status=''COMPLETED''
    ORDER BY CompletedAt DESC,AllocationRunId DESC
) r ON r.AllocationRunId=d.AllocationRunId;
');

EXEC(N'
CREATE OR ALTER VIEW ml.vw_RewardAllocationSummaryCurrent AS
SELECT r.*
FROM ml.RewardAllocationRun r
JOIN (
    SELECT TOP(1) AllocationRunId
    FROM ml.RewardAllocationRun WHERE Status=''COMPLETED''
    ORDER BY CompletedAt DESC,AllocationRunId DESC
) current_run ON current_run.AllocationRunId=r.AllocationRunId;
');
