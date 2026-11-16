select
    cast(user_id     as varchar) as user_id,
    cast(signup_date as date)    as signup_date,
    cast(cohort_week as date)    as cohort_week,
    cast(device      as varchar) as device,
    cast(channel     as varchar) as channel,
    cast(country     as varchar) as country
from {{ source('raw', 'users') }}
