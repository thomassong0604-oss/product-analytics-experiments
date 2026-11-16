-- Sample ratio mismatch: the first thing to check before reading any experiment.
--
-- If assignment was 50/50 but the arms came back 48/52, something upstream is broken — a redirect
-- that drops slow clients, a bot filter that fires unevenly, a logging gap on one variant. When
-- that happens the arms differ in who is in them, not only in what they saw, and every metric below
-- is uninterpretable. The usual rule is to treat p < 0.001 as a stop-and-investigate, deliberately
-- stricter than a metric readout: the cost of ignoring a real mismatch is far higher than the cost
-- of one extra investigation.
--
-- The statistic is compared against fixed critical values rather than converted to a p-value,
-- because DuckDB has no error function and approximating one in SQL would add a second thing to
-- get wrong. With one degree of freedom the thresholds are constants: 3.841 at p = 0.05 and 10.828
-- at p = 0.001. Keeping the check in SQL means anyone with warehouse access can run it.
with counts as (
    select variant, count(*) as users
    from main_marts.fct_user_metrics
    group by variant
),

totals as (
    select sum(users) as total, count(*) as arms from counts
),

expected as (
    select
        c.variant,
        c.users                                as observed,
        t.total::double / t.arms               as expected_users,
        t.total                                as total_users
    from counts c cross join totals t
),

statistic as (
    select
        sum(pow(observed - expected_users, 2) / expected_users) as chi_square,
        count(*) - 1                                            as degrees_of_freedom
    from expected
)

select
    e.variant,
    e.observed,
    round(e.expected_users, 1)                        as expected_users,
    round(e.observed / e.total_users, 5)              as observed_share,
    round(s.chi_square, 4)                            as chi_square,
    s.degrees_of_freedom,
    10.828                                            as critical_value_p001,
    case when s.chi_square > 10.828
              then 'STOP: sample ratio mismatch — investigate before reading any metric'
         when s.chi_square > 3.841
              then 'borderline: imbalanced at p < 0.05, worth a look before reading metrics'
         else 'allocation looks as intended' end      as verdict
from expected e cross join statistic s
order by e.variant
