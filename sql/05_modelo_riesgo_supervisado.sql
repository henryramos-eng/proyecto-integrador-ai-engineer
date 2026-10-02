/* Contrato de persistencia para la etapa 2: modelo supervisado de riesgo. */
SET NOCOUNT ON;
SET XACT_ABORT ON;

IF DB_NAME() <> N'CasinoPalacioReal'
    THROW 51010, 'Este script debe ejecutarse en CasinoPalacioReal.', 1;

IF SCHEMA_ID(N'ml') IS NULL OR OBJECT_ID(N'ml.RiskAnalyticalBase',N'U') IS NULL
    THROW 51011, 'Debe completarse primero la etapa 1 de riesgo.', 1;

IF OBJECT_ID(N'ml.RiskModelRun',N'U') IS NULL
BEGIN
    CREATE TABLE ml.RiskModelRun (
        ModelRunId bigint IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ml_RiskModelRun PRIMARY KEY,
        DataRunId bigint NOT NULL,
        ModelVersion varchar(50) NOT NULL,
        FeatureSetVersion varchar(50) NOT NULL,
        SelectedModel varchar(100) NOT NULL,
        MediumThreshold decimal(12,10) NOT NULL,
        HighThreshold decimal(12,10) NOT NULL,
        TrainRows int NOT NULL,
        ValidationRows int NOT NULL,
        TestRows int NOT NULL,
        MetricsJson nvarchar(max) NOT NULL,
        ArtifactPath nvarchar(1000) NOT NULL,
        Status varchar(20) NOT NULL,
        StartedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_RiskModelRun_StartedAt DEFAULT SYSUTCDATETIME(),
        CompletedAt datetime2(0) NULL,
        CONSTRAINT FK_ml_RiskModelRun_DataRun FOREIGN KEY (DataRunId)
            REFERENCES ml.RiskDataPreparationRun(RunId),
        CONSTRAINT UQ_ml_RiskModelRun_DataVersion UNIQUE (DataRunId,ModelVersion),
        CONSTRAINT CK_ml_RiskModelRun_Status
            CHECK (Status IN ('STARTED','COMPLETED','FAILED')),
        CONSTRAINT CK_ml_RiskModelRun_Thresholds
            CHECK (MediumThreshold BETWEEN 0 AND 1
               AND HighThreshold BETWEEN MediumThreshold AND 1)
    );
END;

IF OBJECT_ID(N'ml.RiskModelPrediction',N'U') IS NULL
BEGIN
    CREATE TABLE ml.RiskModelPrediction (
        ModelRunId bigint NOT NULL,
        FrameId bigint NOT NULL,
        UserID bigint NOT NULL,
        CutoffDate date NOT NULL,
        SplitSet varchar(10) NOT NULL,
        ActualTarget bit NOT NULL,
        RiskProbability decimal(12,10) NOT NULL,
        RiskLevel varchar(10) NOT NULL,
        CreatedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_RiskPrediction_CreatedAt DEFAULT SYSUTCDATETIME(),
        CONSTRAINT PK_ml_RiskModelPrediction PRIMARY KEY (ModelRunId,FrameId),
        CONSTRAINT FK_ml_RiskPrediction_ModelRun FOREIGN KEY (ModelRunId)
            REFERENCES ml.RiskModelRun(ModelRunId),
        CONSTRAINT FK_ml_RiskPrediction_Frame FOREIGN KEY (FrameId)
            REFERENCES ml.RiskAnalyticalBase(FrameId),
        CONSTRAINT CK_ml_RiskPrediction_Split
            CHECK (SplitSet IN ('train','validation','test')),
        CONSTRAINT CK_ml_RiskPrediction_Probability
            CHECK (RiskProbability BETWEEN 0 AND 1),
        CONSTRAINT CK_ml_RiskPrediction_Level
            CHECK (RiskLevel IN ('Bajo','Medio','Alto'))
    );
    CREATE INDEX IX_ml_RiskPrediction_RunSplitLevel
        ON ml.RiskModelPrediction(ModelRunId,SplitSet,RiskLevel);
    CREATE INDEX IX_ml_RiskPrediction_UserCutoff
        ON ml.RiskModelPrediction(UserID,CutoffDate);
END;

EXEC(N'
CREATE OR ALTER VIEW ml.vw_RiskModelPredictionCurrent AS
SELECT p.*
FROM ml.RiskModelPrediction p
JOIN (
    SELECT TOP (1) ModelRunId
    FROM ml.RiskModelRun
    WHERE Status=''COMPLETED''
    ORDER BY CompletedAt DESC,ModelRunId DESC
) r ON r.ModelRunId=p.ModelRunId;
');

