-- A user assigned twice would be counted into both arms, which quietly destroys the comparison
-- while every row count still looks plausible. Any row returned here fails the build.
select user_id, experiment, count(*) as assignments
from {{ ref('stg_assignment') }}
group by user_id, experiment
having count(*) > 1
