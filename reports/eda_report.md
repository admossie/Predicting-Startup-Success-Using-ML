# Startup Dataset EDA & Preparation Report

## 1) Dataset Overview
- Original shape: (10, 20)
- Cleaned shape: (10, 27)
- Duplicates removed (by startup_id): 0

## 2) Missing Values
### Before imputation
- startup_id: 0
- startup_name: 0
- founded_year: 0
- industry: 0
- country: 0
- funding_rounds: 0
- total_funding_usd: 0
- last_funding_stage: 0
- employees: 0
- team_size: 0
- followers: 0
- job_openings: 0
- remote_friendly: 0
- monthly_web_visits: 0
- investor_count: 0
- latest_valuation_usd: 0
- has_exit: 0
- exit_type: 7
- years_to_exit: 7
- startup_age_years: 0

### After imputation
- startup_id: 0
- startup_name: 0
- founded_year: 0
- industry: 0
- country: 0
- funding_rounds: 0
- total_funding_usd: 0
- last_funding_stage: 0
- employees: 0
- team_size: 0
- followers: 0
- job_openings: 0
- remote_friendly: 0
- monthly_web_visits: 0
- investor_count: 0
- latest_valuation_usd: 0
- has_exit: 0
- exit_type: 0
- years_to_exit: 0
- startup_age_years: 0
- years_to_exit_missing_flag: 0
- exit_type_missing_flag: 0
- funding_stage_rank: 0
- log1p_total_funding_usd: 0
- log1p_latest_valuation_usd: 0
- log1p_monthly_web_visits: 0
- log1p_followers: 0

## 3) Outlier Treatment (IQR Capping)
- total_funding_usd: outliers_before=1, outliers_after=0, bounds=(-8125000.00, 19275000.00)
- employees: outliers_before=0, outliers_after=0, bounds=(-82.50, 227.50)
- team_size: outliers_before=0, outliers_after=0, bounds=(-80.62, 218.38)
- followers: outliers_before=0, outliers_after=0, bounds=(300.00, 8700.00)
- job_openings: outliers_before=1, outliers_after=0, bounds=(-0.12, 10.88)
- monthly_web_visits: outliers_before=0, outliers_after=0, bounds=(-8375.00, 82625.00)
- investor_count: outliers_before=0, outliers_after=0, bounds=(-4.38, 22.62)
- latest_valuation_usd: outliers_before=1, outliers_after=0, bounds=(-75625000.00, 193375000.00)
- years_to_exit: outliers_before=0, outliers_after=0, bounds=(-7.88, 13.12)

## 4) Numeric Summary
```
       founded_year  funding_rounds  total_funding_usd  employees  team_size  followers  job_openings  remote_friendly  monthly_web_visits  investor_count  latest_valuation_usd  has_exit  years_to_exit  startup_age_years  years_to_exit_missing_flag  exit_type_missing_flag  funding_stage_rank  log1p_total_funding_usd  log1p_latest_valuation_usd  log1p_monthly_web_visits  log1p_followers
count        10.000          10.000       1.000000e+01     10.000     10.000     10.000        10.000           10.000              10.000          10.000          1.000000e+01    10.000         10.000             10.000                      10.000                  10.000              10.000                   10.000                      10.000                    10.000           10.000
mean       2017.400           2.800       6.567500e+06     78.000     72.800   4500.000         5.488            0.700           37400.000           9.400          6.673750e+07     0.300          2.400              8.600                       0.700                   0.700               1.800                   15.183                      17.527                    10.421            8.334
std           2.221           1.317       5.881091e+06     60.026     56.352   1777.639         2.606            0.483           17883.574           5.147          5.879621e+07     0.483          3.893              2.221                       0.483                   0.483               1.317                    1.235                       1.201                     0.502            0.432
min        2014.000           1.000       5.000000e+05     12.000     10.000   1800.000         2.000            0.000           14000.000           2.000          5.000000e+06     0.000          0.000              5.000                       0.000                   0.000               0.000                   13.122                      15.425                     9.547            7.496
25%        2016.000           2.000       2.150000e+06     33.750     31.500   3450.000         4.000            0.250           25750.000           5.750          2.525000e+07     0.000          0.000              7.250                       0.250                   0.250               1.000                   14.547                      17.023                    10.155            8.144
50%        2017.500           3.000       5.650000e+06     65.000     60.000   4300.000         5.000            1.000           33500.000           9.500          5.250000e+07     0.000          0.000              8.500                       1.000                   1.000               2.000                   15.526                      17.747                    10.417            8.366
75%        2018.750           3.750       9.000000e+06    111.250    106.250   5550.000         6.750            1.000           48500.000          12.500          9.250000e+07     0.750          5.250             10.000                       1.000                   1.000               2.750                   16.008                      18.342                    10.789            8.621
max        2021.000           5.000       1.927500e+07    200.000    185.000   7800.000        10.875            1.000           73000.000          18.000          1.933750e+08     1.000          9.000             12.000                       1.000                   1.000               4.000                   16.774                      19.080                    11.198            8.962
```

## 5) Categorical Summary
```
                   count unique       top freq
startup_id            10     10      S001    1
startup_name          10     10   AlphaAI    1
industry              10     10        Ai    1
country               10      8       Usa    3
last_funding_stage    10      5  Series A    3
exit_type             10      3   No Exit    7
```

## 6) Correlation with has_exit (numeric features)
- years_to_exit_missing_flag: -1.0000
- exit_type_missing_flag: -1.0000
- years_to_exit: 0.9926
- team_size: 0.8719
- employees: 0.8660
- total_funding_usd: 0.8242
- latest_valuation_usd: 0.8144
- funding_stage_rank: 0.8037
- funding_rounds: 0.8037
- investor_count: 0.7955
- founded_year: -0.7456
- startup_age_years: 0.7456
- log1p_total_funding_usd: 0.6690
- log1p_latest_valuation_usd: 0.6651
- job_openings: 0.5659
- remote_friendly: -0.5238
- monthly_web_visits: 0.1775
- log1p_monthly_web_visits: 0.1377
- log1p_followers: 0.1298
- followers: 0.0647