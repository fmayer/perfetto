--
-- Copyright 2020 The Android Open Source Project
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     https://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

select ts, dur, process.pid as pid, display_frame_token, surface_frame_token, layer_name,
    present_type, on_time_finish, gpu_composition, jank_type
from actual_frame_timeline_slice
join process_track on process_track.upid = actual_frame_timeline_slice.upid
join process on process_track.upid = process.upid
where process_track.name = 'Actual Timeline'
order by ts
