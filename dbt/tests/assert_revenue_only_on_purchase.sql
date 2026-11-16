-- Revenue attached to a non-purchase event would inflate every revenue metric downstream.
select event_id, event_name, revenue
from {{ ref('stg_events') }}
where revenue <> 0 and event_name <> 'purchase'
