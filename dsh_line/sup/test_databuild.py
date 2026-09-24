# sup/test_databuild.py
from sup.databuild import align_turns, label_segments

def test_align_turns_perfect_match():
    turns = [{"spk": "a", "text": "我想点一杯珍珠奶茶"}, {"spk": "b", "text": "好的请问甜度"}]
    sentences = [{"start": 1000, "end": 3000, "text": "呃我想点一杯珍珠奶茶"},
                 {"start": 3500, "end": 6000, "text": "好的请问奶茶需要什么"}]
    r = align_turns(turns, sentences)
    assert r[0][0] == 0 and r[1][0] == 1

def test_align_turns_low_similarity_masked():
    turns = [{"spk": "a", "text": "完全无关的一句话"}]
    sentences = [{"start": 0, "end": 2000, "text": "欢迎光临想喝点什么"}]
    assert align_turns(turns, sentences)[0][0] is None

def test_label_segments_max_overlap():
    vsegs = [[1000, 3000], [3000, 5000]]
    seg_ids = [0, 1]
    spk = {0: 0, 1: 1}
    sentences = [{"start": 1000, "end": 3000}, {"start": 3000, "end": 5000}]
    lab = label_segments(vsegs, seg_ids, spk, sentences)
    assert lab == {0: 0, 1: 1}
