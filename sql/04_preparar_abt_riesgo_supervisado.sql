/*
Etapa 1 - Preparacion de datos para el modelo supervisado de riesgo.

Fuente: ext.TransparencyDemographics, ext.TransparencyDailyAggregates y
ext.TransparencyRGEvents.

Contrato temporal:
  - unidad: cliente + fecha de corte;
  - observacion: 90 dias, incluida la fecha de corte;
  - objetivo: primer evento RG durante los 30 dias posteriores;
  - ninguna actividad posterior al corte participa en las variables.

EventTypeFirst=2 representa apelaciones de intervenciones anteriores. La fecha
registrada no es un inicio de riesgo valido; esos clientes se preservan en la
tabla de rechazos y no se usan para entrenar.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

IF DB_NAME() <> N'CasinoPalacioReal'
    THROW 51000, 'Este script debe ejecutarse en CasinoPalacioReal.', 1;

IF SCHEMA_ID(N'ext') IS NULL
    THROW 51001, 'No existe el esquema ext con los datos externos.', 1;

IF SCHEMA_ID(N'ml') IS NULL
    EXEC(N'CREATE SCHEMA ml AUTHORIZATION dbo;');

IF OBJECT_ID(N'ml.RiskDataPreparationRun', N'U') IS NULL
BEGIN
    CREATE TABLE ml.RiskDataPreparationRun (
        RunId bigint IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ml_RiskDataPreparationRun PRIMARY KEY,
        PipelineVersion varchar(50) NOT NULL,
        SourceFingerprint char(64) NOT NULL,
        ObservationDays smallint NOT NULL,
        HorizonDays smallint NOT NULL,
        FirstCutoffDate date NOT NULL,
        SnapshotCount smallint NOT NULL,
        StartedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_RiskRun_StartedAt DEFAULT SYSUTCDATETIME(),
        CompletedAt datetime2(0) NULL,
        Status varchar(20) NOT NULL,
        AcceptedRows bigint NULL,
        RejectedRows bigint NULL,
        Notes nvarchar(2000) NULL,
        CONSTRAINT CK_ml_RiskRun_Status
            CHECK (Status IN ('STARTED', 'COMPLETED', 'FAILED')),
        CONSTRAINT UQ_ml_RiskRun_VersionSource
            UNIQUE (PipelineVersion, SourceFingerprint)
    );
END;

IF OBJECT_ID(N'ml.RiskAnalyticalBase', N'U') IS NULL
BEGIN
    CREATE TABLE ml.RiskAnalyticalBase (
        FrameId bigint IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ml_RiskAnalyticalBase PRIMARY KEY,
        RunId bigint NOT NULL,
        UserID bigint NOT NULL,
        CutoffDate date NOT NULL,
        ObservationStartDate date NOT NULL,
        HorizonEndDate date NOT NULL,
        SplitSet varchar(10) NOT NULL,
        TargetRGEvent bit NOT NULL,

        -- Variables de contexto para auditoria de equidad; no son predictores v1.
        CountryName nvarchar(100) NULL,
        LanguageName nvarchar(100) NULL,
        Gender nvarchar(30) NULL,
        AgeAtCutoff smallint NULL,
        DaysSinceRegistration int NULL,
        DaysSinceFirstDeposit int NULL,

        ActivityRows int NOT NULL,
        ActiveDays smallint NOT NULL,
        ProductCount smallint NOT NULL,
        MonetaryObservedRows int NOT NULL,
        MissingMonetaryPct decimal(9,6) NOT NULL,
        MissingBetsRows int NOT NULL,

        NumberOfBetsTotal bigint NOT NULL,
        AvgBetsPerActiveDay decimal(19,4) NOT NULL,
        MaxDailyBets bigint NOT NULL,
        StdDailyBets decimal(19,4) NULL,
        NumberOfBetsLast30 bigint NOT NULL,
        NumberOfBetsPrevious60 bigint NOT NULL,
        BetsTrendRatio decimal(19,6) NULL,

        TurnoverTotal decimal(19,4) NULL,
        AvgDailyTurnover decimal(19,4) NULL,
        MaxDailyTurnover decimal(19,4) NULL,
        StdDailyTurnover decimal(19,4) NULL,
        TurnoverLast30 decimal(19,4) NULL,
        TurnoverPrevious60 decimal(19,4) NULL,
        TurnoverTrendRatio decimal(19,6) NULL,

        NetHoldTotal decimal(19,4) NULL,
        GrossLossTotal decimal(19,4) NULL,
        GrossWinTotal decimal(19,4) NULL,
        MaxDailyLoss decimal(19,4) NULL,
        StdDailyHold decimal(19,4) NULL,
        LossDays smallint NOT NULL,

        ActiveDaysLast30 smallint NOT NULL,
        ActiveDaysPrevious60 smallint NOT NULL,
        ActiveDaysTrendRatio decimal(19,6) NULL,
        FixedOddsRows int NOT NULL,
        LiveActionRows int NOT NULL,
        PokerRows int NOT NULL,
        LiveActionShare decimal(9,6) NOT NULL,
        PokerShare decimal(9,6) NOT NULL,
        FixedToLiveTurnoverRatio decimal(19,6) NULL,

        CreatedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_RiskABT_CreatedAt DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_ml_RiskABT_Run FOREIGN KEY (RunId)
            REFERENCES ml.RiskDataPreparationRun(RunId),
        CONSTRAINT UQ_ml_RiskABT_RunUserCutoff UNIQUE (RunId, UserID, CutoffDate),
        CONSTRAINT CK_ml_RiskABT_Split CHECK (SplitSet IN ('train','validation','test')),
        CONSTRAINT CK_ml_RiskABT_Dates CHECK (
            ObservationStartDate = DATEADD(day,-89,CutoffDate)
            AND HorizonEndDate = DATEADD(day,30,CutoffDate)
        ),
        CONSTRAINT CK_ml_RiskABT_MissingPct CHECK (MissingMonetaryPct BETWEEN 0 AND 1)
    );

    CREATE INDEX IX_ml_RiskABT_RunSplitTarget
        ON ml.RiskAnalyticalBase(RunId, SplitSet, TargetRGEvent);
    CREATE INDEX IX_ml_RiskABT_RunUser
        ON ml.RiskAnalyticalBase(RunId, UserID, CutoffDate);
END;

IF OBJECT_ID(N'ml.RiskDataRejection', N'U') IS NULL
BEGIN
    CREATE TABLE ml.RiskDataRejection (
        RejectionId bigint IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_ml_RiskDataRejection PRIMARY KEY,
        RunId bigint NOT NULL,
        UserID bigint NOT NULL,
        CutoffDate date NULL,
        ReasonCode varchar(80) NOT NULL,
        ReasonDetail nvarchar(500) NOT NULL,
        CreatedAt datetime2(0) NOT NULL
            CONSTRAINT DF_ml_RiskReject_CreatedAt DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_ml_RiskReject_Run FOREIGN KEY (RunId)
            REFERENCES ml.RiskDataPreparationRun(RunId)
    );
    CREATE INDEX IX_ml_RiskReject_RunReason
        ON ml.RiskDataRejection(RunId, ReasonCode);
END;

IF OBJECT_ID(N'ml.RiskDataQualitySummary', N'U') IS NULL
BEGIN
    CREATE TABLE ml.RiskDataQualitySummary (
        RunId bigint NOT NULL,
        MetricName varchar(100) NOT NULL,
        MetricValue decimal(28,8) NULL,
        Status varchar(10) NOT NULL,
        Detail nvarchar(1000) NULL,
        CONSTRAINT PK_ml_RiskDataQualitySummary PRIMARY KEY (RunId, MetricName),
        CONSTRAINT FK_ml_RiskQuality_Run FOREIGN KEY (RunId)
            REFERENCES ml.RiskDataPreparationRun(RunId),
        CONSTRAINT CK_ml_RiskQuality_Status CHECK (Status IN ('PASS','WARN','FAIL'))
    );
END;

DECLARE @PipelineVersion varchar(50) = 'risk-abt-v1.0.0';
DECLARE @ObservationDays smallint = 90;
DECLARE @HorizonDays smallint = 30;
DECLARE @FirstCutoffDate date = '2008-11-01';
DECLARE @SnapshotCount smallint = 14;
DECLARE @SourcePayload nvarchar(max);
DECLARE @SourceFingerprint char(64);
DECLARE @RunId bigint;

SELECT @SourcePayload = STRING_AGG(
    CONVERT(nvarchar(max), CONCAT(DatasetKey, ':', SHA256, ':', RowsInserted)), N'|'
) WITHIN GROUP (ORDER BY DatasetKey)
FROM ext.DatasetIngestionLog
WHERE DatasetKey IN (
    'transparency_demographics',
    'transparency_daily_aggregates',
    'transparency_rg_events'
)
AND Status = 'COMPLETED';

IF (SELECT COUNT(*) FROM ext.DatasetIngestionLog WHERE DatasetKey IN (
        'transparency_demographics',
        'transparency_daily_aggregates',
        'transparency_rg_events'
    ) AND Status='COMPLETED') <> 3
    THROW 51002, 'Faltan fuentes Transparency completadas en ext.DatasetIngestionLog.', 1;

SET @SourceFingerprint = CONVERT(char(64), HASHBYTES('SHA2_256', @SourcePayload), 2);

SELECT @RunId = RunId
FROM ml.RiskDataPreparationRun
WHERE PipelineVersion = @PipelineVersion
  AND SourceFingerprint = @SourceFingerprint
  AND Status = 'COMPLETED';

IF @RunId IS NOT NULL
BEGIN
    EXEC(N'
    CREATE OR ALTER VIEW ml.vw_RiskAnalyticalBaseCurrent AS
    SELECT a.*
    FROM ml.RiskAnalyticalBase a
    JOIN (
        SELECT TOP (1) RunId
        FROM ml.RiskDataPreparationRun
        WHERE Status=''COMPLETED''
        ORDER BY CompletedAt DESC,RunId DESC
    ) r ON r.RunId=a.RunId;
    ');
    EXEC(N'
    CREATE OR ALTER VIEW ml.vw_RiskDataQualityCurrent AS
    SELECT q.*
    FROM ml.RiskDataQualitySummary q
    JOIN (
        SELECT TOP (1) RunId
        FROM ml.RiskDataPreparationRun
        WHERE Status=''COMPLETED''
        ORDER BY CompletedAt DESC,RunId DESC
    ) r ON r.RunId=q.RunId;
    ');
    SELECT RunId, PipelineVersion, SourceFingerprint, AcceptedRows, RejectedRows, Status
    FROM ml.RiskDataPreparationRun
    WHERE RunId = @RunId;
    PRINT 'SKIP: la misma version y las mismas fuentes ya fueron procesadas.';
    RETURN;
END;

SELECT @RunId = RunId
FROM ml.RiskDataPreparationRun
WHERE PipelineVersion=@PipelineVersion AND SourceFingerprint=@SourceFingerprint;

IF @RunId IS NULL
BEGIN
    INSERT INTO ml.RiskDataPreparationRun (
        PipelineVersion, SourceFingerprint, ObservationDays, HorizonDays,
        FirstCutoffDate, SnapshotCount, Status, Notes
    )
    VALUES (
        @PipelineVersion, @SourceFingerprint, @ObservationDays, @HorizonDays,
        @FirstCutoffDate, @SnapshotCount, 'STARTED',
        N'Panel cliente-fecha; split por cliente; EventTypeFirst=2 excluido por fecha de inicio no fiable.'
    );
    SET @RunId = SCOPE_IDENTITY();
END
ELSE
BEGIN
    DELETE FROM ml.RiskDataQualitySummary WHERE RunId=@RunId;
    DELETE FROM ml.RiskDataRejection WHERE RunId=@RunId;
    DELETE FROM ml.RiskAnalyticalBase WHERE RunId=@RunId;
    UPDATE ml.RiskDataPreparationRun
    SET StartedAt=SYSUTCDATETIME(),CompletedAt=NULL,Status='STARTED',
        AcceptedRows=NULL,RejectedRows=NULL,
        Notes=N'Reintento idempotente: panel cliente-fecha; EventTypeFirst=2 excluido.'
    WHERE RunId=@RunId;
END;

BEGIN TRY
    BEGIN TRANSACTION;

    CREATE TABLE #Cutoffs (CutoffDate date NOT NULL PRIMARY KEY);
    INSERT INTO #Cutoffs(CutoffDate)
    SELECT DATEADD(day, 30*v.n, @FirstCutoffDate)
    FROM (VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(11),(12),(13)) v(n);

    SELECT
        d.UserID,
        c.CutoffDate,
        DATEADD(day, -(@ObservationDays-1), c.CutoffDate) AS ObservationStartDate,
        DATEADD(day, @HorizonDays, c.CutoffDate) AS HorizonEndDate,
        CASE
            WHEN ABS(CONVERT(bigint, CHECKSUM(d.UserID))) % 100 < 70 THEN 'train'
            WHEN ABS(CONVERT(bigint, CHECKSUM(d.UserID))) % 100 < 85 THEN 'validation'
            ELSE 'test'
        END AS SplitSet,
        CONVERT(bit, CASE
            WHEN r.RGFirstDate > c.CutoffDate
             AND r.RGFirstDate <= DATEADD(day, @HorizonDays, c.CutoffDate)
            THEN 1 ELSE 0 END) AS TargetRGEvent,
        d.CountryName,
        d.LanguageName,
        d.Gender,
        d.YearOfBirth,
        d.RegistrationDate,
        d.FirstDepositDate
    INTO #CandidateFrames
    FROM ext.TransparencyDemographics d
    CROSS JOIN #Cutoffs c
    LEFT JOIN ext.TransparencyRGEvents r ON r.UserID = d.UserID
    WHERE ISNULL(r.EventTypeFirst, -1) <> 2
      AND (r.RGFirstDate IS NULL OR r.RGFirstDate > c.CutoffDate);

    CREATE UNIQUE CLUSTERED INDEX IX_TempCandidate
        ON #CandidateFrames(UserID, CutoffDate);

    INSERT INTO ml.RiskDataRejection(RunId, UserID, CutoffDate, ReasonCode, ReasonDetail)
    SELECT @RunId, d.UserID, NULL, 'UNRELIABLE_EVENT_ONSET',
           N'EventTypeFirst=2 corresponde a una apelacion de una intervencion RG anterior; la fecha no representa el inicio del riesgo.'
    FROM ext.TransparencyDemographics d
    JOIN ext.TransparencyRGEvents r ON r.UserID=d.UserID
    WHERE r.EventTypeFirst=2;

    INSERT INTO ml.RiskDataRejection(RunId, UserID, CutoffDate, ReasonCode, ReasonDetail)
    SELECT @RunId, f.UserID, f.CutoffDate, 'NO_ACTIVITY_IN_LOOKBACK',
           N'No existe actividad observada durante los 90 dias anteriores o en la fecha de corte.'
    FROM #CandidateFrames f
    WHERE NOT EXISTS (
        SELECT 1
        FROM ext.TransparencyDailyAggregates a
        WHERE a.UserID=f.UserID
          AND a.ActivityDate >= f.ObservationStartDate
          AND a.ActivityDate < DATEADD(day,1,f.CutoffDate)
    );

    SELECT
        f.UserID,
        f.CutoffDate,
        CAST(a.ActivityDate AS date) AS ActivityDate,
        COUNT(*) AS SourceRows,
        SUM(CASE WHEN a.Turnover IS NOT NULL THEN 1 ELSE 0 END) AS MonetaryObservedRows,
        SUM(CASE WHEN a.Turnover IS NULL THEN 1 ELSE 0 END) AS MissingMonetaryRows,
        SUM(CASE WHEN a.NumberOfBets IS NULL THEN 1 ELSE 0 END) AS MissingBetsRows,
        SUM(COALESCE(a.NumberOfBets,0)) AS DailyBets,
        SUM(a.Turnover) AS DailyTurnover,
        SUM(a.Hold) AS DailyHold
    INTO #FrameDaily
    FROM #CandidateFrames f
    JOIN ext.TransparencyDailyAggregates a
      ON a.UserID=f.UserID
     AND a.ActivityDate >= f.ObservationStartDate
     AND a.ActivityDate < DATEADD(day,1,f.CutoffDate)
    GROUP BY f.UserID, f.CutoffDate, CAST(a.ActivityDate AS date);

    CREATE UNIQUE CLUSTERED INDEX IX_TempFrameDaily
        ON #FrameDaily(UserID, CutoffDate, ActivityDate);

    SELECT
        f.UserID,
        f.CutoffDate,
        COUNT(DISTINCT a.ProductType) AS ProductCount,
        SUM(CASE WHEN a.ProductType=1 THEN 1 ELSE 0 END) AS FixedOddsRows,
        SUM(CASE WHEN a.ProductType=2 THEN 1 ELSE 0 END) AS LiveActionRows,
        SUM(CASE WHEN a.ProductType=10 THEN 1 ELSE 0 END) AS PokerRows,
        SUM(CASE WHEN a.ProductType=1 THEN a.Turnover END) AS FixedOddsTurnover,
        SUM(CASE WHEN a.ProductType=2 THEN a.Turnover END) AS LiveActionTurnover
    INTO #FrameProduct
    FROM #CandidateFrames f
    JOIN ext.TransparencyDailyAggregates a
      ON a.UserID=f.UserID
     AND a.ActivityDate >= f.ObservationStartDate
     AND a.ActivityDate < DATEADD(day,1,f.CutoffDate)
    GROUP BY f.UserID, f.CutoffDate;

    CREATE UNIQUE CLUSTERED INDEX IX_TempFrameProduct
        ON #FrameProduct(UserID, CutoffDate);

    WITH DailyAgg AS (
        SELECT
            d.UserID,
            d.CutoffDate,
            COUNT(*) AS ActiveDays,
            SUM(d.SourceRows) AS ActivityRows,
            SUM(d.MonetaryObservedRows) AS MonetaryObservedRows,
            SUM(d.MissingMonetaryRows) AS MissingMonetaryRows,
            SUM(d.MissingBetsRows) AS MissingBetsRows,
            SUM(d.DailyBets) AS NumberOfBetsTotal,
            MAX(d.DailyBets) AS MaxDailyBets,
            STDEV(CONVERT(float,d.DailyBets)) AS StdDailyBets,
            SUM(CASE WHEN d.ActivityDate>=DATEADD(day,-29,d.CutoffDate) THEN d.DailyBets ELSE 0 END) AS NumberOfBetsLast30,
            SUM(CASE WHEN d.ActivityDate< DATEADD(day,-29,d.CutoffDate) THEN d.DailyBets ELSE 0 END) AS NumberOfBetsPrevious60,
            SUM(d.DailyTurnover) AS TurnoverTotal,
            AVG(d.DailyTurnover) AS AvgDailyTurnover,
            MAX(d.DailyTurnover) AS MaxDailyTurnover,
            STDEV(CONVERT(float,d.DailyTurnover)) AS StdDailyTurnover,
            SUM(CASE WHEN d.ActivityDate>=DATEADD(day,-29,d.CutoffDate) THEN d.DailyTurnover END) AS TurnoverLast30,
            SUM(CASE WHEN d.ActivityDate< DATEADD(day,-29,d.CutoffDate) THEN d.DailyTurnover END) AS TurnoverPrevious60,
            SUM(d.DailyHold) AS NetHoldTotal,
            SUM(CASE WHEN d.DailyHold>0 THEN d.DailyHold ELSE 0 END) AS GrossLossTotal,
            SUM(CASE WHEN d.DailyHold<0 THEN -d.DailyHold ELSE 0 END) AS GrossWinTotal,
            MAX(CASE WHEN d.DailyHold>0 THEN d.DailyHold ELSE 0 END) AS MaxDailyLoss,
            STDEV(CONVERT(float,d.DailyHold)) AS StdDailyHold,
            SUM(CASE WHEN d.DailyHold>0 THEN 1 ELSE 0 END) AS LossDays,
            SUM(CASE WHEN d.ActivityDate>=DATEADD(day,-29,d.CutoffDate) THEN 1 ELSE 0 END) AS ActiveDaysLast30,
            SUM(CASE WHEN d.ActivityDate< DATEADD(day,-29,d.CutoffDate) THEN 1 ELSE 0 END) AS ActiveDaysPrevious60,
            MAX(d.ActivityDate) AS MaxObservedActivityDate
        FROM #FrameDaily d
        GROUP BY d.UserID, d.CutoffDate
    )
    INSERT INTO ml.RiskAnalyticalBase (
        RunId, UserID, CutoffDate, ObservationStartDate, HorizonEndDate,
        SplitSet, TargetRGEvent, CountryName, LanguageName, Gender,
        AgeAtCutoff, DaysSinceRegistration, DaysSinceFirstDeposit,
        ActivityRows, ActiveDays, ProductCount, MonetaryObservedRows,
        MissingMonetaryPct, MissingBetsRows,
        NumberOfBetsTotal, AvgBetsPerActiveDay, MaxDailyBets, StdDailyBets,
        NumberOfBetsLast30, NumberOfBetsPrevious60, BetsTrendRatio,
        TurnoverTotal, AvgDailyTurnover, MaxDailyTurnover, StdDailyTurnover,
        TurnoverLast30, TurnoverPrevious60, TurnoverTrendRatio,
        NetHoldTotal, GrossLossTotal, GrossWinTotal, MaxDailyLoss,
        StdDailyHold, LossDays, ActiveDaysLast30, ActiveDaysPrevious60,
        ActiveDaysTrendRatio, FixedOddsRows, LiveActionRows, PokerRows,
        LiveActionShare, PokerShare, FixedToLiveTurnoverRatio
    )
    SELECT
        @RunId,
        f.UserID,
        f.CutoffDate,
        f.ObservationStartDate,
        f.HorizonEndDate,
        f.SplitSet,
        f.TargetRGEvent,
        f.CountryName,
        f.LanguageName,
        f.Gender,
        CASE WHEN f.YearOfBirth IS NOT NULL
                  AND YEAR(f.CutoffDate)-f.YearOfBirth BETWEEN 18 AND 100
             THEN YEAR(f.CutoffDate)-f.YearOfBirth END,
        CASE WHEN f.RegistrationDate<=f.CutoffDate
             THEN DATEDIFF(day,f.RegistrationDate,f.CutoffDate) END,
        CASE WHEN f.FirstDepositDate<=f.CutoffDate
             THEN DATEDIFF(day,f.FirstDepositDate,f.CutoffDate) END,
        a.ActivityRows,
        a.ActiveDays,
        p.ProductCount,
        a.MonetaryObservedRows,
        CONVERT(decimal(9,6), a.MissingMonetaryRows*1.0/NULLIF(a.ActivityRows,0)),
        a.MissingBetsRows,
        a.NumberOfBetsTotal,
        CONVERT(decimal(19,4),a.NumberOfBetsTotal*1.0/NULLIF(a.ActiveDays,0)),
        a.MaxDailyBets,
        CONVERT(decimal(19,4),a.StdDailyBets),
        a.NumberOfBetsLast30,
        a.NumberOfBetsPrevious60,
        CONVERT(decimal(19,6),a.NumberOfBetsLast30*2.0/NULLIF(a.NumberOfBetsPrevious60,0)),
        CONVERT(decimal(19,4),a.TurnoverTotal),
        CONVERT(decimal(19,4),a.AvgDailyTurnover),
        CONVERT(decimal(19,4),a.MaxDailyTurnover),
        CONVERT(decimal(19,4),a.StdDailyTurnover),
        CONVERT(decimal(19,4),a.TurnoverLast30),
        CONVERT(decimal(19,4),a.TurnoverPrevious60),
        CONVERT(decimal(19,6),a.TurnoverLast30*2.0/NULLIF(a.TurnoverPrevious60,0)),
        CONVERT(decimal(19,4),a.NetHoldTotal),
        CONVERT(decimal(19,4),a.GrossLossTotal),
        CONVERT(decimal(19,4),a.GrossWinTotal),
        CONVERT(decimal(19,4),a.MaxDailyLoss),
        CONVERT(decimal(19,4),a.StdDailyHold),
        a.LossDays,
        a.ActiveDaysLast30,
        a.ActiveDaysPrevious60,
        CONVERT(decimal(19,6),a.ActiveDaysLast30*2.0/NULLIF(a.ActiveDaysPrevious60,0)),
        p.FixedOddsRows,
        p.LiveActionRows,
        p.PokerRows,
        CONVERT(decimal(9,6),p.LiveActionRows*1.0/NULLIF(a.ActivityRows,0)),
        CONVERT(decimal(9,6),p.PokerRows*1.0/NULLIF(a.ActivityRows,0)),
        CONVERT(decimal(19,6),p.FixedOddsTurnover*1.0/NULLIF(p.LiveActionTurnover,0))
    FROM #CandidateFrames f
    JOIN DailyAgg a ON a.UserID=f.UserID AND a.CutoffDate=f.CutoffDate
    JOIN #FrameProduct p ON p.UserID=f.UserID AND p.CutoffDate=f.CutoffDate;

    DECLARE @AcceptedRows bigint = (
        SELECT COUNT_BIG(*) FROM ml.RiskAnalyticalBase WHERE RunId=@RunId
    );
    DECLARE @RejectedRows bigint = (
        SELECT COUNT_BIG(*) FROM ml.RiskDataRejection WHERE RunId=@RunId
    );
    DECLARE @PositiveRows bigint = (
        SELECT COUNT_BIG(*) FROM ml.RiskAnalyticalBase WHERE RunId=@RunId AND TargetRGEvent=1
    );

    INSERT INTO ml.RiskDataQualitySummary(RunId,MetricName,MetricValue,Status,Detail)
    SELECT @RunId,'accepted_rows',@AcceptedRows,
           CASE WHEN @AcceptedRows>0 THEN 'PASS' ELSE 'FAIL' END,
           N'Filas cliente-fecha con actividad en la ventana de observacion'
    UNION ALL SELECT @RunId,'positive_rows',@PositiveRows,
           CASE WHEN @PositiveRows>0 THEN 'PASS' ELSE 'FAIL' END,
           N'Primer evento RG observado en los 30 dias posteriores al corte'
    UNION ALL SELECT @RunId,'target_positive_rate',
           @PositiveRows*1.0/NULLIF(@AcceptedRows,0),'PASS',
           N'Desbalance esperado; se tratara en modelado, sin sobremuestreo en la ABT'
    UNION ALL SELECT @RunId,'distinct_users',COUNT(DISTINCT UserID),'PASS',
           N'Clientes unicos aceptados' FROM ml.RiskAnalyticalBase WHERE RunId=@RunId
    UNION ALL SELECT @RunId,'rejected_no_activity_frames',COUNT_BIG(*),'WARN',
           N'Ventanas conservadas en tabla de rechazo; no se imputo actividad inexistente'
           FROM ml.RiskDataRejection WHERE RunId=@RunId AND ReasonCode='NO_ACTIVITY_IN_LOOKBACK'
    UNION ALL SELECT @RunId,'excluded_unreliable_onset_users',COUNT_BIG(*),'WARN',
           N'Apelaciones previas (EventTypeFirst=2), excluidas por fuga/etiqueta temporal no fiable'
           FROM ml.RiskDataRejection WHERE RunId=@RunId AND ReasonCode='UNRELIABLE_EVENT_ONSET'
    UNION ALL SELECT @RunId,'raw_duplicate_business_keys',COUNT_BIG(*),'WARN',
           N'Claves UserID+fecha+producto repetidas; se agregaron, no se eliminaron silenciosamente'
           FROM (
               SELECT UserID,CAST(ActivityDate AS date) d,ProductType
               FROM ext.TransparencyDailyAggregates
               GROUP BY UserID,CAST(ActivityDate AS date),ProductType
               HAVING COUNT_BIG(*)>1
           ) q
    UNION ALL SELECT @RunId,'raw_missing_monetary_rows',COUNT_BIG(*),'WARN',
           N'Turnover/Hold faltantes por transferencia de terceros; se preservan como NULL'
           FROM ext.TransparencyDailyAggregates WHERE Turnover IS NULL OR Hold IS NULL
    UNION ALL SELECT @RunId,'future_activity_leak_rows',COUNT_BIG(*),
           CASE WHEN COUNT_BIG(*)=0 THEN 'PASS' ELSE 'FAIL' END,
           N'Debe ser cero: actividad agregada posterior a la fecha de corte'
           FROM #FrameDaily WHERE ActivityDate>CutoffDate
    UNION ALL SELECT @RunId,'duplicate_abt_keys',COUNT_BIG(*),
           CASE WHEN COUNT_BIG(*)=0 THEN 'PASS' ELSE 'FAIL' END,
           N'Debe ser cero: duplicados RunId+UserID+CutoffDate'
           FROM (
               SELECT UserID,CutoffDate
               FROM ml.RiskAnalyticalBase WHERE RunId=@RunId
               GROUP BY UserID,CutoffDate HAVING COUNT_BIG(*)>1
           ) q;

    IF EXISTS (
        SELECT 1 FROM ml.RiskDataQualitySummary
        WHERE RunId=@RunId AND Status='FAIL'
    )
        THROW 51003, 'La ABT no supero los controles criticos de calidad.', 1;

    UPDATE ml.RiskDataPreparationRun
    SET CompletedAt=SYSUTCDATETIME(), Status='COMPLETED',
        AcceptedRows=@AcceptedRows, RejectedRows=@RejectedRows
    WHERE RunId=@RunId;

    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    IF @@TRANCOUNT>0 ROLLBACK TRANSACTION;
    UPDATE ml.RiskDataPreparationRun
    SET CompletedAt=SYSUTCDATETIME(), Status='FAILED',
        Notes=CONCAT(Notes,N' | ',ERROR_MESSAGE())
    WHERE RunId=@RunId;
    THROW;
END CATCH;

EXEC(N'
CREATE OR ALTER VIEW ml.vw_RiskAnalyticalBaseCurrent AS
SELECT a.*
FROM ml.RiskAnalyticalBase a
JOIN (
    SELECT TOP (1) RunId
    FROM ml.RiskDataPreparationRun
    WHERE Status=''COMPLETED''
    ORDER BY CompletedAt DESC,RunId DESC
) r ON r.RunId=a.RunId;
');

EXEC(N'
CREATE OR ALTER VIEW ml.vw_RiskDataQualityCurrent AS
SELECT q.*
FROM ml.RiskDataQualitySummary q
JOIN (
    SELECT TOP (1) RunId
    FROM ml.RiskDataPreparationRun
    WHERE Status=''COMPLETED''
    ORDER BY CompletedAt DESC,RunId DESC
) r ON r.RunId=q.RunId;
');

SELECT RunId,PipelineVersion,SourceFingerprint,ObservationDays,HorizonDays,
       FirstCutoffDate,SnapshotCount,AcceptedRows,RejectedRows,Status
FROM ml.RiskDataPreparationRun
WHERE RunId=@RunId;

SELECT SplitSet,TargetRGEvent,COUNT_BIG(*) AS Rows,COUNT(DISTINCT UserID) AS Users
FROM ml.RiskAnalyticalBase
WHERE RunId=@RunId
GROUP BY SplitSet,TargetRGEvent
ORDER BY SplitSet,TargetRGEvent;

SELECT MetricName,MetricValue,Status,Detail
FROM ml.RiskDataQualitySummary
WHERE RunId=@RunId
ORDER BY CASE Status WHEN 'FAIL' THEN 1 WHEN 'WARN' THEN 2 ELSE 3 END,MetricName;

