-- Shift marker/region positions without moving media, changing tempo or rerunning Studio.
-- Positive seconds = later; negative seconds = earlier. One Ctrl+Z undoes the shift.

local title = "Shift lyric markers / chapter regions"
local function message(text)
  reaper.ShowMessageBox(text, title, 0)
end

local accepted, input = reaper.GetUserInputs(title, 3,
  "Seconds (+ later / - earlier),Scope (all / selected),Include regions (yes / no)",
  "0,all,yes")
if not accepted then return end

local amount, scope, include = input:match("^%s*([^,]+)%s*,%s*([^,]+)%s*,%s*([^,]+)%s*$")
local function trim(value) return value and value:match("^%s*(.-)%s*$"):lower() end
local seconds = tonumber(amount)
scope, include = trim(scope), trim(include)
if not seconds or seconds ~= seconds or math.abs(seconds) == math.huge
    or (scope ~= "all" and scope ~= "selected")
    or (include ~= "yes" and include ~= "no") then
  message("Enter a finite offset in seconds, scope all or selected, and regions yes or no.")
  return
end
if seconds == 0 then return end
if scope == "selected" and (not reaper.GetRegionOrMarker or not reaper.GetRegionOrMarkerInfo_Value) then
  message("Selected-only shifts need a recent REAPER version. Your REAPER 7.83 supports this option.")
  return
end

local targets, index = {}, 0
while true do
  local found, is_region, position, ending, name, number, color = reaper.EnumProjectMarkers3(0, index)
  if found == 0 then break end
  local selected = scope == "all"
  if not selected then
    local handle = reaper.GetRegionOrMarker(0, index, "")
    selected = handle and reaper.GetRegionOrMarkerInfo_Value(0, handle, "B_UISEL") ~= 0
  end
  if selected and (not is_region or include == "yes") then
    if position + seconds < 0 then
      message("This shift would put a marker or region before project time zero. Nothing was moved.")
      return
    end
    targets[#targets + 1] = {region=is_region, position=position + seconds,
      ending=is_region and ending + seconds or position + seconds,
      name=name, number=number, color=color}
  end
  index = index + 1
end
if #targets == 0 then
  message("No matching markers/regions. For selected scope, select their rows in Region/Marker Manager first.")
  return
end

local changed = 0
reaper.Undo_BeginBlock2(0)
reaper.PreventUIRefresh(1)
local ok, error_text = xpcall(function()
  for _, target in ipairs(targets) do
    local moved = reaper.SetProjectMarker3(0, target.number, target.region,
      target.position, target.ending, target.name, target.color)
    if not moved then error("REAPER could not move marker/region " .. target.number) end
    changed = changed + 1
  end
end, debug.traceback)
reaper.PreventUIRefresh(-1)
reaper.Undo_EndBlock2(0, string.format("Shift %d markers/regions by %+.3f seconds", changed, seconds), -1)
reaper.UpdateArrange()
if not ok then
  if changed > 0 then reaper.Undo_DoUndo2(0) end
  message("The shift failed. Successful changes from this attempt were undone.\n" .. tostring(error_text))
end
