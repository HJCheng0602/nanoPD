# engine/tests/test_scheduler.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from block_manager.sequence import Sequence, SequenceGroup, SequenceStatus
from block_manager.block_manager import BlockSpaceManager
from engine.scheduler import Scheduler

def make_group(seq_id: int, prompt_len: int, block_size: int) -> SequenceGroup:
    """Build a dummy SequenceGroup; token content doesn't matter."""
    token_ids = list(range(prompt_len))  # arbitrary ids; content doesn't matter
    seq = Sequence(seq_id=seq_id, prompt_token_ids=token_ids, block_size=block_size)
    return SequenceGroup(request_id=str(seq_id), seqs=[seq])

def test_scheduler():
    block_size = 16
    num_blocks = 64
    block_manager = BlockSpaceManager(block_size=block_size, num_gpu_blocks=num_blocks)
    scheduler = Scheduler(block_manager=block_manager, max_batch_size=3)

    # add three requests to waiting queue
    g0 = make_group(seq_id=0, prompt_len=10, block_size=block_size)
    g1 = make_group(seq_id=1, prompt_len=20, block_size=block_size)
    g2 = make_group(seq_id=2, prompt_len=5,  block_size=block_size)
    scheduler.waiting.extend([g0, g1, g2])

    for step in range(6):
        output = scheduler.schedule()

        prefill_name = output.prefill_group.request_id if output.prefill_group else "None"
        decode_names = [g.request_id for g in output.decode_groups]
        print(f"step {step} | prefill: {prefill_name} | decode: {decode_names}")

        # simulate engine behaviour: add to running after prefill completes
        if output.prefill_group:
            # block_manager.allocate(output.prefill_group)  # uncomment to test real block allocation under memory pressure
            for seq in output.prefill_group.get_seqs():
                seq.status = SequenceStatus.RUNNING
            scheduler.running.append(output.prefill_group)

        # simulate decode: mark g0 as finished at step 3
        if step == 3:
            for seq in g0.get_seqs():
                seq.status = SequenceStatus.FINISHED_STOPPED
            print(f"  → g0 marked as finished")

if __name__ == "__main__":
    test_scheduler()