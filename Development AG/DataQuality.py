import pandas as pd



def ana_encoding(ana):

    ana_encoded = ana[["participant", "age", "sex", "education", "work", "fracture-type", "ethnicity"]].copy()

    # male=1, female=0
    ana_encoded["sex_male"] = ana_encoded["sex"].map({"male": 1, "female": 0})

    # Education label encoding with doctorate as highest level
    education_order = {
        "secondary education": 0,
        "undergraduate degree": 1,
        "graduate degree": 2,
        "doctorate degree": 3
    }
    ana_encoded["education_label"] = ana_encoded["education"].map(education_order)

    # retired=0, employed part-time=1
    ana_encoded["work_part_time"] = ana_encoded["work"].map({"retired": 0, "employed part-time": 1})

    fracture_dummies = pd.get_dummies(
    ana_encoded["fracture-type"],
    prefix="fracture",
    dtype=int
    )

    ethnicity_dummies = pd.get_dummies(
    ana_encoded["ethnicity"],
    prefix="ethnicity",
    dtype=int
    )

    num_ana = pd.concat(
    [
        ana_encoded[["participant", "age", "sex_male", "education_label", "work_part_time"]],
        fracture_dummies,
        ethnicity_dummies,
    ],
    axis=1,
    )
    
    return num_ana



def initial_check(df, num_ana, data):

    # 1. Shape
    print("\n--- Shape ---")
    n_participants = df['participant'].nunique()
    rows_per_part  = df.shape[0] // n_participants
    print(f"  features  : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"              {n_participants} participants x {rows_per_part} days per participant")
    print(f"  ana      : {num_ana.shape[0]} rows x {num_ana.shape[1]} cols")

    # 2. Missing values
    print("\n--- Missingness ---")
    feat_null_total = df.isna().sum().sum()
    demo_null_total = num_ana.isna().sum().sum()
    print(f"  features.csv  : {feat_null_total} total NaN values")
    print(f"  demographics.csv  : {demo_null_total} total NaN values")

    feat_null_cols = df.columns[df.isna().any()].tolist()
    feat_null_cols

    if feat_null_cols:
        print(f"  [WARN] Columns with nulls in features: {feat_null_cols}")
        print(df[feat_null_cols].isna().sum().to_string())
    else:
        print("  [OK] No missing values in features")

    demo_null_cols = num_ana.columns[num_ana.isna().any()].tolist()
    if demo_null_cols:
        print(f"  [WARN] Columns with nulls in demographics: {demo_null_cols}")
    else:
        print("  [OK] No missing values in demographics")

    # 3. Duplicates
    print("\n--- Duplicate Keys ---")
    dup_full = df.duplicated().sum()   ### tiene la prima occorrenza come originale (False); le copie successive sono True.
    dup_key  = df.duplicated(subset=['participant', 'timestamp']).sum()
    dup_demo = num_ana.duplicated(subset=['participant']).sum()
    print(f"  features full-row duplicates           : {dup_full}")
    print(f"  features (participant, timestamp) dupes: {dup_key}")
    print(f"  demographics participant duplicates    : {dup_demo}")

    assert dup_key  == 0, f"Duplicate (participant, timestamp) keys: {dup_key}"
    assert dup_demo == 0, f"Duplicate participant rows in demographics: {dup_demo}"

    # 4. Participant balance
    print("\n--- Participant Balance ---")
    rows_by_part = data.groupby('participant').size()
    print(f"  Rows per participant: min={rows_by_part.min()}, max={rows_by_part.max()}, "
      f"all equal={rows_by_part.nunique()==1}")  ### nunique() è un metodo di pandas, conta quanti valori distinti ci sono
    assert rows_by_part.nunique() == 1, f"Participants have unequal row counts: {rows_by_part.to_dict()}"

    # 5. Timestamp ordering
    print("\n--- Timestamp Coverage ---")
    print(f"  Global date range: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
    dup_ts = data.groupby('participant')['timestamp'].apply(lambda x: x.duplicated().sum())  ### somma i timestamp duplicati per ogni partecipante
    assert dup_ts.sum() == 0, f"Duplicate timestamps within participants: {dup_ts[dup_ts>0]}"

    # 6. Target ranges
    print("\n--- Target Distributions ---")
    for t in ['sis', 'ohs', 'oks']:
        s = data[t]
        print(f"  {t}: min={s.min():4} max={s.max():4} mean={s.mean():.2f} "
            f"std={s.std():.2f} nulls={s.isna().sum()}")

    return 0



def merge_check(data, df):

    assert data.shape[0] == df.shape[0], f"Row count changed after merge: {df.shape[0]} -> {data.shape[0]}"
    missing_after_merge = data.isna().sum().sum()
    assert missing_after_merge == 0, f"Missing demographics after merge ({missing_after_merge} NaNs) — participant ID mismatch?"

    print("=== Merge result ===")
    print(f"  df_raw shape    : {df.shape}")
    print(f"  df_merged shape : {data.shape}")
    print()

    return 0



def col_log(COLS_DROP_ALWAYS, SAFE_SENSOR_COLS, SENSOR_MODALITIES, ANA_QUAL_COLS, ANA_QUAN_COLS):
    print("=== Column constants set ===")
    print(f"  COLS_DROP_ALWAYS: {len(COLS_DROP_ALWAYS)} columns")
    print(f"  Safe sensor cols: {len(SAFE_SENSOR_COLS)} across {len(SENSOR_MODALITIES)} modalities")
    for mod, cols in SENSOR_MODALITIES.items():
        print(f"    {mod:<14}: {len(cols)} features")
    print(f"  ANA qualitative cols   : {ANA_QUAL_COLS}")
    print(f"  ANA quantitative cols   : {ANA_QUAN_COLS}")

    return 0



def manage_target(df_invest, data, df_merge, PARTICIPANT_COL, CLINICAL_TS_COL, TARGET_SIS, TARGET_OHS, TARGET_OKS):

    inconsistent_pids = []
    for pid in sorted(df_invest[PARTICIPANT_COL].unique()):
        sub = df_invest[df_invest[PARTICIPANT_COL] == pid]
        grp = sub.groupby(CLINICAL_TS_COL)[[TARGET_SIS, TARGET_OHS, TARGET_OKS]].agg(
            lambda x: x.nunique()  ## nunique() ne conta i distinti
        )
        if (grp > 1).any(axis=1).any():  ## .any(axis=1) guarda ogni riga e restituisce True se in quella riga c’è almeno un True, poi any() guarda alla serie e restituisce True se in quella serie c’è almeno un True
            print(f"  [FLAG] Participant {pid}: unique SIS/OHS/OKS counts per clinical-timestamp block:")
            print(grp.to_string(index=True))
            detail = sub.groupby(CLINICAL_TS_COL)[[TARGET_SIS, TARGET_OHS, TARGET_OKS]].apply(
                lambda g: g[[TARGET_SIS, TARGET_OHS, TARGET_OKS]].drop_duplicates()
            )
            print("     Actual SIS/OHS/OKS values within each block:")
            print(detail.to_string())   #### mostra i valori originali
            print()
            inconsistent_pids.append(pid)

    print(f"Participants with inconsistencies: {inconsistent_pids}")

    # Replace each clinical block with its median target value
    df_resolve = data.copy()
    df_resolve[CLINICAL_TS_COL] = df_merge[CLINICAL_TS_COL]

    for target in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
        block_median = (
            df_resolve
            .groupby([PARTICIPANT_COL, CLINICAL_TS_COL])[target]
            .transform('median')  ## median, which keeps the majority value and avoids using information from other blocks.
        )  ## le parentesi servono solo a spezzare la riga, senza usare \
        df_resolve[target] = block_median.round().astype(int)

    df_resolve = df_resolve.drop(columns=[CLINICAL_TS_COL])

    # Check that each block now has only one target value
    df_verify = df_resolve.copy()
    df_verify[CLINICAL_TS_COL] = df_merge[CLINICAL_TS_COL]

    print("=== Post-Resolution Verification ===")
    all_clean = True
    for pid in sorted(df_verify[PARTICIPANT_COL].unique()):
        sub = df_verify[df_verify[PARTICIPANT_COL] == pid]
        grp = sub.groupby(CLINICAL_TS_COL)[[TARGET_SIS, TARGET_OHS, TARGET_OKS]].nunique()
        if (grp > 1).any(axis=1).any():
            print(f"  [FAIL] P{pid}: still inconsistent after resolution!")
            print(grp)
            all_clean = False

    if all_clean:
        print("  [OK] All {participant, clinical-timestamp} blocks have exactly 1 unique SIS/OHS/OKS value.")

    print("Updated target stats after median-resolution:")
    for t in [TARGET_SIS, TARGET_OHS, TARGET_OKS]:
        s = df_resolve[t]
        print(f"  {t}: min={s.min()} max={s.max()} mean={s.mean():.2f} std={s.std():.2f}")

    df_model = df_resolve.copy()
    print()
    print(f"[OK] df_model shape: {df_model.shape} (rows, cols)")

    return df_model



def final_report(df_model, SAFE_SENSOR_COLS, TARGET_SIS, TARGET_OHS, TARGET_OKS, 
                       SIS_SUBITEMS, OHS_SUBITEMS, OKS_SUBITEMS, CLINICAL_TESTS, CLINICAL_TS_COL,
                       PARTICIPANT_COL, DATE_COL, SENSOR_MODALITIES):

    # Final check before modelling
    FORBIDDEN_IN_MODEL = (
        SIS_SUBITEMS + OHS_SUBITEMS + OKS_SUBITEMS + CLINICAL_TESTS + [CLINICAL_TS_COL]
        )

    ## FORBIDDEN_IN_MODEL == COLS_DROP_ALWAYS  ## True

    leaked = [c for c in FORBIDDEN_IN_MODEL if c in df_model.columns]
    assert len(leaked) == 0, f"LEAKAGE GATE FAILED: {leaked}"
    assert TARGET_SIS in df_model.columns, "TARGET_SIS missing from df_model"
    assert TARGET_OHS in df_model.columns, "TARGET_OHS missing from df_model"
    assert TARGET_OKS in df_model.columns, "TARGET_OKS missing from df_model"
    missing_sensor = [c for c in SAFE_SENSOR_COLS if c not in df_model.columns]
    assert len(missing_sensor) == 0, f"Sensor cols missing: {missing_sensor}"

    print("=== Final Leakage Gate ===")
    print(f"  Forbidden columns in df_model : {leaked} -> PASS")
    print(f"  Targets present               : {TARGET_SIS}, {TARGET_OHS}, {TARGET_OKS} -> PASS")
    print(f"  Sensor cols present           : {len(SAFE_SENSOR_COLS)} / {len(SAFE_SENSOR_COLS)} -> PASS")
    print()
    print("=== df_model Column Inventory ===")
    print(f"  Total columns: {df_model.shape[1]}")
    print(f"  Metadata     : {PARTICIPANT_COL}, {DATE_COL}")
    print(f"  Targets      : {TARGET_SIS}, {TARGET_OHS}, {TARGET_OKS}")
    print(f"  Sensor feats : {len(SAFE_SENSOR_COLS)} across {len(SENSOR_MODALITIES)} modalities")
    for mod, cols in SENSOR_MODALITIES.items():
        print(f"    {mod:<14}: {len(cols)}")
    print()
    print(df_model.head(3))

    return 0