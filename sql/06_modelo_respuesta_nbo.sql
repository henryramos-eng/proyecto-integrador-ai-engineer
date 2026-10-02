/* Contrato de persistencia para etapa 3: respuesta por recompensa. */
SET NOCOUNT ON;
SET XACT_ABORT ON;

IF DB_NAME() <> N'CasinoPalacioReal'
    THROW 51020, 'Este script debe ejecutarse en CasinoPalacioReal.', 1;
IF SCHEMA_ID(N'ml') IS NULL
    EXEC(N'CREATE SCHEMA ml AUTHORIZATION dbo;');

IF OBJECT_ID(N'ml.ResponseModelRun',N'U') IS NULL
BEGIN
    CREATE TABLE ml.ResponseModelRun (
        ResponseModelRunId bigint IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ml_ResponseModelRun PRIMARY KEY,
        ModelVersion varchar(60) NOT NULL,
        SimulationVersion varchar(60) NOT NULL,
        SourceFingerprint char(64) NOT NULL,
        SelectedModel varchar(100) NOT NULL,
        MetricsJson nvarchar(max) NOT NULL,
        ArtifactPath nvarchar(1000) NOT NULL,
        Status varchar(20) NOT NULL,
        StartedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_ResponseRun_StartedAt DEFAULT SYSUTCDATETIME(),
        CompletedAt datetime2(0) NULL,
        CONSTRAINT UQ_ml_ResponseRun_VersionSource UNIQUE(ModelVersion,SourceFingerprint),
        CONSTRAINT CK_ml_ResponseRun_Status CHECK(Status IN ('STARTED','COMPLETED','FAILED'))
    );
END;

IF OBJECT_ID(N'ml.ResponseEvidenceAnchor',N'U') IS NULL
BEGIN
    CREATE TABLE ml.ResponseEvidenceAnchor (
        ResponseModelRunId bigint NOT NULL,
        SourceDataset varchar(100) NOT NULL,
        TreatmentLabel varchar(100) NOT NULL,
        OutcomeName varchar(50) NOT NULL,
        SourceRows bigint NOT NULL,
        Responses bigint NOT NULL,
        ObservedRate decimal(14,12) NOT NULL,
        RoleInSimulation nvarchar(500) NOT NULL,
        CONSTRAINT PK_ml_ResponseEvidenceAnchor
            PRIMARY KEY(ResponseModelRunId,SourceDataset,TreatmentLabel),
        CONSTRAINT FK_ml_ResponseAnchor_Run FOREIGN KEY(ResponseModelRunId)
            REFERENCES ml.ResponseModelRun(ResponseModelRunId)
    );
END;

IF OBJECT_ID(N'ml.ResponseCampaignSynthetic',N'U') IS NULL
BEGIN
    CREATE TABLE ml.ResponseCampaignSynthetic (
        ResponseModelRunId bigint NOT NULL,
        CampaignId varchar(30) NOT NULL,
        OfferDate date NOT NULL,
        IdCliente bigint NOT NULL,
        RewardType varchar(10) NOT NULL,
        SplitSet varchar(10) NOT NULL,
        Responded bit NOT NULL,
        SimulationProbability decimal(12,10) NOT NULL,
        ValueIfResponse decimal(19,6) NOT NULL,
        ActualIncrementalValue decimal(19,6) NOT NULL,
        Cost decimal(19,6) NOT NULL,
        DataNature varchar(60) NOT NULL,
        CONSTRAINT PK_ml_ResponseCampaign
            PRIMARY KEY(ResponseModelRunId,CampaignId,IdCliente),
        CONSTRAINT FK_ml_ResponseCampaign_Run FOREIGN KEY(ResponseModelRunId)
            REFERENCES ml.ResponseModelRun(ResponseModelRunId),
        CONSTRAINT CK_ml_ResponseCampaign_Reward
            CHECK(RewardType IN ('control','baja','media','alta')),
        CONSTRAINT CK_ml_ResponseCampaign_Split
            CHECK(SplitSet IN ('train','validation','test')),
        CONSTRAINT CK_ml_ResponseCampaign_Probability
            CHECK(SimulationProbability BETWEEN 0 AND 1)
    );
    CREATE INDEX IX_ml_ResponseCampaign_RunSplitReward
        ON ml.ResponseCampaignSynthetic(ResponseModelRunId,SplitSet,RewardType);
END;

IF OBJECT_ID(N'ml.ResponseOption',N'U') IS NULL
BEGIN
    CREATE TABLE ml.ResponseOption (
        ResponseModelRunId bigint NOT NULL,
        IdCliente bigint NOT NULL,
        RewardType varchar(10) NOT NULL,
        ControlProbability decimal(12,10) NOT NULL,
        ResponseProbability decimal(12,10) NOT NULL,
        UpliftProbability decimal(12,10) NOT NULL,
        ValueIncremental decimal(19,6) NOT NULL,
        Cost decimal(19,6) NOT NULL,
        ExpectedValue decimal(19,6) NOT NULL,
        IncrementalExpectedValue decimal(19,6) NOT NULL,
        CreatedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_ResponseOption_CreatedAt DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_ml_ResponseOption PRIMARY KEY(ResponseModelRunId,IdCliente,RewardType),
        CONSTRAINT FK_ml_ResponseOption_Run FOREIGN KEY(ResponseModelRunId)
            REFERENCES ml.ResponseModelRun(ResponseModelRunId),
        CONSTRAINT CK_ml_ResponseOption_Reward CHECK(RewardType IN ('baja','media','alta')),
        CONSTRAINT CK_ml_ResponseOption_Probabilities CHECK(
            ControlProbability BETWEEN 0 AND 1 AND ResponseProbability BETWEEN 0 AND 1
        )
    );
    CREATE INDEX IX_ml_ResponseOption_RunRewardValue
        ON ml.ResponseOption(ResponseModelRunId,RewardType,ExpectedValue DESC);
END;

EXEC(N'
CREATE OR ALTER VIEW ml.vw_ResponseOptionCurrent AS
SELECT o.*
FROM ml.ResponseOption o
JOIN (
    SELECT TOP(1) ResponseModelRunId
    FROM ml.ResponseModelRun WHERE Status=''COMPLETED''
    ORDER BY CompletedAt DESC,ResponseModelRunId DESC
) r ON r.ResponseModelRunId=o.ResponseModelRunId;
');

EXEC(N'
CREATE OR ALTER VIEW ml.vw_ResponseCampaignCurrent AS
SELECT c.*
FROM ml.ResponseCampaignSynthetic c
JOIN (
    SELECT TOP(1) ResponseModelRunId
    FROM ml.ResponseModelRun WHERE Status=''COMPLETED''
    ORDER BY CompletedAt DESC,ResponseModelRunId DESC
) r ON r.ResponseModelRunId=c.ResponseModelRunId;
');

