"""A parallel event stream keeps every active node visible and counted once."""
import os,types
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['POINTYOINK_NO_DEVICE']='1'
import app2
from PySide6.QtWidgets import QApplication
from app2.builder.stagehost import apply_event
from app2.builder.progress_words import headline
from app2.qt.build_step import BuildStepPage,run_overview
from app2.qt.build_view import ProgressCard
from app2.qt.bridge import SessionBridge

def stream():
    prog=types.SimpleNamespace(nodes={},text='',current=None,frac=0.,plan={},elapsed=lambda:60)
    ids=['side.fuse@3','side.fuse@4','side.platter@2','side.keep@2','fused','report']
    apply_event('plan',dict(nodes=[dict(id=i,status='stale') for i in ids],to_run=6,cached=0,est_s=300),prog)
    for nid,n in zip(ids[:3],(3,4,2)):apply_event('node',dict(id=nid,state='start',n=n),prog)
    apply_event('progress',dict(id=ids[0],frac=.5,overall=.25,done=5,total=10,text='frame 5 of 10'),prog)
    apply_event('progress',dict(id=ids[1],frac=.75,overall=.5,done=15,total=20,text='frame 15 of 20'),prog)
    return prog

def test_headline_groups_all_running_sides_and_page_uses_it():
    app=QApplication.instance() or QApplication([]);prog=stream();page=BuildStepPage(SessionBridge(),False);page.prog=prog
    text='Sides 3 and 4: Fusion, Side 2: Cut base'
    assert headline(prog.nodes)==text and text in prog.text
    assert page._what(['side.fuse@3','side.fuse@4','side.platter@2'])[0]==text
    page.shutdown();page.close()

def test_card_has_one_updating_row_per_node_and_aggregate_percent():
    app=QApplication.instance() or QApplication([]);prog=stream();card=ProgressCard()
    card.show_state('Working','last node','',prog.frac,60,0,nodes=prog.nodes,overall=run_overview(prog)[0])
    assert card.title.text()=='Sides 3 and 4: Fusion, Side 2: Cut base' and card.bar.value()==500
    assert len(card._events)==3 and 'frame 5 of 10' in card.log.toPlainText() and 'frame 15 of 20' in card.log.toPlainText()
    apply_event('node',dict(id='side.fuse@3',state='done',secs=14,n=3),prog)
    card.show_state('Working','last node','',prog.frac,62,0,nodes=prog.nodes,overall=run_overview(prog)[0])
    assert len(card._events)==3 and 'done in 14 s' in card.log.toPlainText()
    assert 'Side 3:' not in card.title.text() and 'Side 4: Fusion' in card.title.text()
    assert '1 of 6 steps done, 2 running' in card.overall.text() and '1 min left' in card.overall.text()
    card.close()

def test_retry_is_counted_and_completed_nodes_leave_active_headline():
    prog=stream();prog.plan['nodes'].append(dict(id='retry',status='warn',retry=True))
    apply_event('node',dict(id='side.fuse@4',state='done',secs=14,n=4),prog)
    apply_event('node',dict(id='side.fuse@3',state='done',secs=14,n=3),prog)
    assert 'Side 2: Cut base'==headline(prog.nodes) and prog.current=='side.platter@2'
    assert 'of 7' in run_overview(prog)[0]
