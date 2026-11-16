-- One row per event, typed and de-duplicated.
--
-- Event streams arrive with duplicates: a retried client request, a replayed Kafka partition, an
-- at-least-once delivery guarantee. De-duplicating here rather than in every downstream model means
-- a double-counted purchase cannot reach a metric, and it keeps the rule in one place where it can
-- be tested.
with typed as (
    select
        cast(event_id    as varchar)   as event_id,
        cast(user_id     as varchar)   as user_id,
        cast(session_id  as varchar)   as session_id,
        cast(event_name  as varchar)   as event_name,
        cast(funnel_step as integer)   as funnel_step,
        cast(event_ts    as timestamp) as event_ts,
        cast(revenue     as double)    as revenue
    from {{ source('raw', 'events') }}
),

deduplicated as (
    select *
    from typed
    -- Keep the first arrival of each event id. row_number rather than distinct, so a duplicate that
    -- disagrees on a column still collapses to exactly one row instead of two.
    qualify row_number() over (partition by event_id order by event_ts) = 1
)

select
    event_id,
    user_id,
    session_id,
    event_name,
    funnel_step,
    event_ts,
    cast(event_ts as date) as event_date,
    date_trunc('week', event_ts) as event_week,
    revenue
from deduplicated
