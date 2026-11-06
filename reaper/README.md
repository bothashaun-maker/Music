# Shift lyric markers in REAPER

[Download Shift_markers_and_regions.lua](https://raw.githubusercontent.com/bothashaun-maker/Music/main/reaper/Shift_markers_and_regions.lua)

One-time installation:

1. Download the `.lua` file. If your browser displays its code, use **Ctrl+S** and
   keep the `.lua` extension (not `.txt`).
2. In REAPER choose **Actions → Show action list…**.
3. Choose **New action → Load ReaScript…** and select that file.
4. Select **Script: Shift_markers_and_regions.lua** and click **Run**.

Enter the offset in seconds: `-0.25` moves earlier by a quarter second; `0.25`
moves later. Choose `all` or `selected`, then `yes` to include chapter regions
or `no` for markers only. Selected scope uses the selected rows in Region/Marker
Manager; REAPER 7.83 supports the necessary native selection API.

For a uniform offset, choose `all` and include regions to keep chapter boundaries
with the lyrics. For one passage, select its marker rows before running with
`selected`. A fixed offset preserves spacing; it cannot repair varying alignment
errors or missing repeated lines. Media items, tempo and audio remain untouched.
Ctrl+Z undoes the action. An offset crossing project time zero is rejected before
anything moves. Marker names, numbers, colors and region lengths are preserved.

The Lua action has automated mocked-REAPER checks; it has not been run in the
user's live REAPER project. It does not require SWS, ReaPack or Timestamp Studio.
