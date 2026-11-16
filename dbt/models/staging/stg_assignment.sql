-- One row per user per experiment. A user appearing twice would silently double-count them into
-- both arms, so the grain is asserted in schema.yml rather than assumed.
select
    cast(user_id     as varchar)   as user_id,
    cast(experiment  as varchar)   as experiment,
    cast(variant     as varchar)   as variant,
    cast(assigned_at as timestamp) as assigned_at
from {{ source('raw', 'assignment') }}
