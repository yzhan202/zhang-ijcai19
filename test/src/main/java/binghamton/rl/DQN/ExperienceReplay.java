package binghamton.rl.DQN;

import java.util.*;
import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.jfree.util.Log;


public class ExperienceReplay {
	
	final int batchSize;
	final private Random random;
	
	private CircularFifoQueue<StateActTrans> storage;
	
	public ExperienceReplay(int maxSize, int batchSize, Random rdm) {
		this.batchSize = batchSize;
		this.random = rdm;
		storage = new CircularFifoQueue<>(maxSize);
	}
	
	public ArrayList<StateActTrans> getBatch(int size) {
		
//		Set<Integer> intSet = new HashSet<>();
		ArrayList<Integer> intList = new ArrayList<Integer>();
		int storageSize = storage.size();
		while(intList.size() < size) {
			int rd = random.nextInt(storageSize);
			intList.add(rd);
		}
		ArrayList<StateActTrans> batch = new ArrayList<>(size);
		Iterator<Integer> iter = intList.iterator();
		while (iter.hasNext()) {
			StateActTrans trans = storage.get(iter.next());
			batch.add(trans.dup());
		}
		
		return batch;
	}
	
	public ArrayList<StateActTrans> getBatch() {
		return getBatch(batchSize);
	}
	
	public void store(StateActTrans transition) {
		storage.add(transition);
	}
	
	public List<StateActTrans> getLastTrajectory() {
		List<StateActTrans> trajectory = new ArrayList<StateActTrans>();
		int storageSize = storage.size();
		trajectory.add(storage.get(storageSize-1).dup());
		int idx = storageSize-2;
		while(idx>=0) {
			StateActTrans trans = storage.get(idx);
			idx--;
			if (trans.isTerminal) {
				break;
			}
			trajectory.add(trans.dup());
		}
		return trajectory;
	}
	
	public void clear() {
		storage.clear();
	}
}

