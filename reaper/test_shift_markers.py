"""Run the shipped Lua action with a mocked REAPER API (requires lupa)."""
from pathlib import Path
import unittest

from lupa import LuaRuntime

SCRIPT = Path(__file__).with_name('Shift_markers_and_regions.lua').read_text()
HARNESS = r'''
records = {
 {region=false,position=1,ending=0,name="First lyric",number=1,color=12,selected=false},
 {region=false,position=3,ending=0,name="Second lyric",number=2,color=34,selected=true},
 {region=true,position=2,ending=6,name="Chorus",number=1,color=56,selected=true}
}
media_position = 0
undo_count, refresh, writes = 0, 0, 0
messages = {}
local snapshot
local function copy(source)
 local out={}
 for i,r in ipairs(source) do out[i]={}; for k,v in pairs(r) do out[i][k]=v end end
 return out
end
reaper = {
 ShowMessageBox=function(text) messages[#messages+1]=text end,
 GetUserInputs=function() return accepted,input end,
 EnumProjectMarkers3=function(_,index)
  local r=records[index+1]
  if not r then return 0 end
  return index+1,r.region,r.position,r.ending,r.name,r.number,r.color
 end,
 GetRegionOrMarker=function(_,index) return index+1 end,
 GetRegionOrMarkerInfo_Value=function(_,handle) return records[handle].selected and 1 or 0 end,
 Undo_BeginBlock2=function() snapshot=copy(records) end,
 Undo_EndBlock2=function() undo_count=undo_count+1 end,
 Undo_DoUndo2=function() records=copy(snapshot) end,
 PreventUIRefresh=function(delta) refresh=refresh+delta end,
 UpdateArrange=function() end,
 SetProjectMarker3=function(_,number,region,position,ending,name,color)
  if fail_number and number==fail_number and not region then return false end
  for _,r in ipairs(records) do
   if r.number==number and r.region==region then
    r.position,r.ending,r.name,r.color=position,ending,name,color
    writes=writes+1
    return true
   end
  end
  return false
 end
}
'''


class ShiftTests(unittest.TestCase):
    def run_action(self, value, accepted=True, fail_number=None):
        lua = LuaRuntime()
        lua.execute(HARNESS)
        state = lua.globals()
        state.input, state.accepted, state.fail_number = value, accepted, fail_number
        lua.execute(SCRIPT)
        self.assertEqual(state.media_position, 0)
        self.assertEqual(state.refresh, 0)
        return state

    def test_shift_all_preserves_spacing_region_length_and_metadata(self):
        state = self.run_action('0.25,all,yes')
        self.assertEqual([state.records[i].position for i in (1,2,3)], [1.25,3.25,2.25])
        self.assertEqual(state.records[3].ending, 6.25)
        self.assertEqual(state.records[3].ending-state.records[3].position, 4)
        self.assertEqual(state.records[1].name, 'First lyric')
        self.assertEqual(state.records[1].color, 12)
        self.assertEqual(state.records[1].number, 1)
        self.assertEqual(state.undo_count, 1)

    def test_shift_selected_marker_earlier_leaves_other_rows_alone(self):
        state = self.run_action('-0.15, selected, no')
        self.assertEqual(state.records[1].position, 1)
        self.assertEqual(state.records[2].position, 2.85)
        self.assertEqual(state.records[3].position, 2)
        self.assertEqual(state.writes, 1)

    def test_selected_regions_can_shift_with_selected_markers(self):
        state = self.run_action('2,selected,yes')
        self.assertEqual(state.records[1].position, 1)
        self.assertEqual(state.records[2].position, 5)
        self.assertEqual(state.records[3].position, 4)
        self.assertEqual(state.records[3].ending, 8)

    def test_crossing_zero_rejects_every_change(self):
        state = self.run_action('-2,all,yes')
        self.assertEqual(state.writes, 0)
        self.assertEqual(state.undo_count, 0)
        self.assertIn('Nothing was moved', state.messages[1])

    def test_cancel_and_zero_are_no_ops(self):
        for value, accepted in [('1,all,yes',False), ('0,all,yes',True)]:
            with self.subTest(value=value, accepted=accepted):
                state = self.run_action(value, accepted)
                self.assertEqual(state.writes, 0)
                self.assertEqual(state.undo_count, 0)

    def test_invalid_offset_and_scope_are_rejected(self):
        for value in ('not-a-number,all,yes','1e309,all,yes','1,typo,yes','1,all,typo','1'):
            with self.subTest(value=value):
                state = self.run_action(value)
                self.assertEqual(state.writes, 0)
                self.assertEqual(len(state.messages), 1)

    def test_partial_failure_rolls_back_successful_moves(self):
        state = self.run_action('1,all,yes', fail_number=2)
        self.assertEqual([state.records[i].position for i in (1,2,3)], [1,3,2])
        self.assertIn('were undone', state.messages[1])


if __name__ == '__main__':
    unittest.main()
