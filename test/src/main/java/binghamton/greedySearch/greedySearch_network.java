package binghamton.greedySearch;

import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.WeightLearningApplication;
import edu.umd.cs.psl.application.learning.weight.em.HardEM;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.util.GroundKernels;
import edu.umd.cs.psl.application.util.Grounding;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.config.ConfigManager;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ResultList;
import edu.umd.cs.psl.evaluation.result.FullInferenceResult;
import edu.umd.cs.psl.evaluation.statistics.RankingScore;
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.PersistedAtomManager;
import edu.umd.cs.psl.model.atom.RandomVariableAtom;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel;
import edu.umd.cs.psl.model.kernel.Kernel;
import edu.umd.cs.psl.model.parameters.PositiveWeight;
import edu.umd.cs.psl.model.parameters.Weight;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;
import edu.umd.cs.psl.reasoner.admm.ADMMReasoner;
import edu.umd.cs.psl.util.database.Queries;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;

import com.google.common.collect.Iterables;

import java.util.Stack;


public class greedySearch_network {
	
	final StandardPredicate[] X;
	final StandardPredicate[] Y;
	final StandardPredicate[] Z;
	final StandardPredicate[] allPosPreds;
	final StandardPredicate[] allNegPreds;
	final StandardPredicate[] negFeatPreds;
	final StandardPredicate[] friendPreds;
	
	final StandardPredicate UserDummyPred;
	final StandardPredicate FriendDummyPred;
	
	final Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap;
	final Map<StandardPredicate, List<List<Object>>> specificPredArgsMap;
	
	final int maxRuleLen;
	final int maxRuleNum;
	final int posPredNum;
	final int negPredNum;
	
	PSLModel model;
	DataStore data;
	Database wlTruthDB;
	Partition trainPart;
	
	final double LAMBDA_ruleLen = 0.0;
	final double LAMBDA_ruleNum = 0.0;
	final double LAMBDA_coverage = 0.0;
	final double LAMBDA_diversity = 0.0;
	final double LAMBDA_semantic = 0.0;
	final double MIN_VAL = 1e-6;
	
	final int outputSize;
	final int ReturnAction;
	int[][] ruleListEmbedding;
	
	private ConfigBundle config;
	final RankingScore[] metrics = new RankingScore[] {RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC};
	final double[] LAMBDA_AUC = new double[] {0, 0, 1.0};
	
	public greedySearch_network(StandardPredicate[] X, StandardPredicate[] Y, StandardPredicate[] Z, 
			PSLModel model, DataStore data, Database wlTruthDB, Partition trainPart, ConfigBundle config,
			Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap, Map<StandardPredicate, List<List<Object>>> specificPredArgsMap,
			StandardPredicate[] negFeatPreds, StandardPredicate[] friendPreds) {
		this.maxRuleLen = 5; 
		this.maxRuleNum = 15; //8;
		
		this.X = X;
		this.Y = Y;
		this.Z = Z;
		this.negFeatPreds = negFeatPreds;
		this.friendPreds = friendPreds;
		this.allPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(this.X, this.Y), this.Z);
		this.allNegPreds = ArrayUtils.addAll(ArrayUtils.addAll(negFeatPreds, this.Y), this.Z); //allPosPreds.clone();
		this.generalPredArgsMap = generalPredArgsMap;
		this.specificPredArgsMap = specificPredArgsMap;
		this.posPredNum = allPosPreds.length;
		this.negPredNum = allNegPreds.length;
		
		this.UserDummyPred = (StandardPredicate)PredicateFactory.getFactory().getPredicate("attendsAA");
		this.FriendDummyPred = (StandardPredicate)PredicateFactory.getFactory().getPredicate("friends");
		
		this.outputSize = posPredNum+ negPredNum+ 1;
		this.ReturnAction = outputSize-1;
		ruleListEmbedding = new int[maxRuleNum][outputSize];
		
		this.model = model;
		this.data = data;
		this.wlTruthDB = wlTruthDB;
		this.trainPart = trainPart;
		this.config = config;	
	}
	
	public void search() throws ClassNotFoundException, IllegalAccessException, InstantiationException {		
		double globalScore = 0;
		int userDummyIdx = getDummyUserPredIdx();
		int friendDummyIdx = getDummyFriendPredIdx();
		Set<StandardPredicate> friendPredSet = new HashSet<StandardPredicate>(Arrays.asList(friendPreds));

		for (int i=0; i<maxRuleNum; i++) {
			int[][] rule_target = new int[2][outputSize];
			double[] score = new double[2];
			for (int j=0; j<2; j++) {
				rule_target[j][posPredNum-1+ negPredNum*j]=1;
				rule_target[j][userDummyIdx] = 1;
				score[j] = 0;
				
				while(true) {
					int optimalAction = -1;
					
					for (int k=0; k<outputSize; k++) {
						if (rule_target[j][k]==1 || k==(posPredNum-1) || k==(posPredNum+negPredNum-1) || k==userDummyIdx || k==friendDummyIdx)
							continue;
						
						rule_target[j][k] = 1;
						for (int r=0; r<posPredNum; r++) {
							if (rule_target[j][r]==1)
								if (friendPredSet.contains(allPosPreds[r]))
									rule_target[j][friendDummyIdx] = 1;
						}
						
						double tmpScore = buildPSLRule(rule_target[j]);
						rule_target[j][k] = 0;
						rule_target[j][friendDummyIdx] = 0;
						// Delete current rule from PSL model, clean up previous rules
						removeLastRuleAndResetModel();
						
						if (tmpScore >= score[j]) {
							score[j] = tmpScore;
							optimalAction = k;
						}
					}
					rule_target[j][optimalAction] = 1;
					for (int r=0; r<posPredNum; r++) {
						if (rule_target[j][r]==1)
							if (friendPredSet.contains(allPosPreds[r]))
								rule_target[j][friendDummyIdx] = 1;
					}
					if (optimalAction == ReturnAction) {
						System.out.println("Rule ["+ i+","+ j+"] Return Action");
						break;
					}
					if (optimalAction < posPredNum)
						System.out.println("Rule ["+ i+","+ j+"] Optimal Action "+ allPosPreds[optimalAction]+ ", "+ score[j]);
					else
						System.out.println("Rule ["+ i+ ","+j+"] Optimal Action ~"+ allNegPreds[optimalAction-posPredNum]+ ", "+ score[j]);
				}
			}
			int targetStar=0;
			for (int r = 0; r < score.length; r++) {
				targetStar = score[r] > score[targetStar] ? r : targetStar;
			}
			double tmpScore = score[targetStar];
			if (tmpScore > globalScore) {
				globalScore = tmpScore;
				ruleCopy(rule_target[targetStar], ruleListEmbedding[i]);
				// Add Optimal Rule to PSL Model, clean up previous rules
				addRuleAndResetModel(ruleListEmbedding[i]);
				System.out.println("Temporal Optimal "+ globalScore+ ", "+ model.toString());
			} else {
				// Reset PSL Model
				AUCcost();
				break;
			}
		}
		
		System.out.println("\nOptimal Result: "+ globalScore+ ", "+ model.toString());
	}
	
	public int getDummyUserPredIdx() {
		for (int r=0; r<posPredNum; r++) {
			StandardPredicate p = allPosPreds[r];
			if (p==UserDummyPred) {
				return r;
			}
		}
		return -1;
	}
	
	public int getDummyFriendPredIdx() {
		for (int r=0; r<posPredNum; r++) {
			StandardPredicate p = allPosPreds[r];
			if (p==FriendDummyPred) {
				return r;
			}
		}
		return -1;
	}
	
	public int getNegPredIdxFromPosPredIdx(int posPredIdx) {
		StandardPredicate posPred = allPosPreds[posPredIdx];
		for (int i=0; i<negPredNum; i++) {
			if (allNegPreds[i] == posPred) {
				return i;
			}
		}
		return -1;
	}
	
	public void ruleCopy(int[] fromArr, int[] toArr) {
		int size = fromArr.length;
		IntStream.range(0, size).forEach(r-> toArr[r]=fromArr[r]);
	}
	
	public void removeLastRuleAndResetModel() {
		double initW = 5.0;
		List<Kernel> kernelList = new ArrayList<>();
		for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)) {
			Weight newWeight = new PositiveWeight(initW);
			k.setWeight(newWeight);
			kernelList.add(k);
		}
		model.removeKernel(kernelList.get(kernelList.size()-1));
	}
	
	public void addRuleAndResetModel(int[] ruleEmbedding) throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		double initW = 5.0;
		for (CompatibilityKernel k : Iterables.filter(model.getKernels(), CompatibilityKernel.class)) {
			Weight newWeight = new PositiveWeight(initW);
			k.setWeight(newWeight);
		}
		buildPSLRule(ruleEmbedding);
	}
	
//	public boolean checkIsNetworkRule(int[] lastRulePreds) {	
//		Set<StandardPredicate> friendPredSet = new HashSet<StandardPredicate>(Arrays.asList(friendPreds));
//		for (int i=0; i<posPredNum; i++) {
//			if (lastRulePreds[i]!=0) {
//				if (friendPredSet.contains(allPosPreds[i]))
//					return true;
//			}
//		}
//		return false;
//	}
	
	public double AUCcost() throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		// Define Partition and Databases
		Partition targetPart = new Partition(2);
		Partition inferenceWritePart = new Partition(3);
		
		Set<StandardPredicate> closedPredicates = new HashSet<StandardPredicate>(Arrays.asList(X));
		Database wlTrainDB = data.getDatabase(targetPart, closedPredicates, trainPart);
		Database inferenceDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
		
		ResultList allGroundings = wlTruthDB.executeQuery(Queries.getQueryForAllAtoms(Y[0]));
		for (int i=0; i<allGroundings.size(); i++) {
			GroundTerm[] grounding = allGroundings.get(i);
			GroundAtom inferAtom = inferenceDB.getAtom(Y[0], grounding);
			GroundAtom wlAtom = wlTrainDB.getAtom(Y[0], grounding);
			if (inferAtom instanceof RandomVariableAtom) {
				inferenceDB.commit((RandomVariableAtom) inferAtom);
			}
			if (wlAtom instanceof RandomVariableAtom) {
				wlTrainDB.commit((RandomVariableAtom) wlAtom);
			}
		}
		
		// Do Weight Learning
		WeightLearningApplication weightLearner = null;
		if (Z.length>0) {
			weightLearner = new HardEM(model, wlTrainDB, wlTruthDB, config);
		} else {
			weightLearner = new MaxLikelihoodMPE(model, wlTrainDB, wlTruthDB, config);
		}
		weightLearner.learn();
		weightLearner.close();
		wlTrainDB.close();
		data.deletePartition(targetPart);
		
//		double loss_grounded = (numGroundedList.stream().mapToInt(Integer::intValue).sum())*1.0/numGroundedList.size();
		
		// Do Inference
		MPEInference mpe = new MPEInference(model, inferenceDB, config);
		FullInferenceResult result = mpe.mpeInference();
		inferenceDB.close();
		
		Set<StandardPredicate> InferredSet = new HashSet<StandardPredicate>(Arrays.asList(Y));
		Database resultDB = data.getDatabase(inferenceWritePart, InferredSet);
		SimpleRankingComparator comparator = new SimpleRankingComparator(resultDB);
		comparator.setBaseline(wlTruthDB);
		double[] score =  new double[metrics.length];
		for (int r=0; r<metrics.length; r++) {
			comparator.setRankingScore(metrics[r]);
			score[r] = comparator.compare(Y[0]);
		}
		resultDB.close();
		data.deletePartition(inferenceWritePart);
		
		double reward_auc = 0;
		for (int i=0; i<metrics.length; i++) {
			reward_auc += score[i]* LAMBDA_AUC[i];
		}
		if (Double.isNaN(reward_auc)) {
			reward_auc = MIN_VAL;
		}
		double reward = reward_auc;
//		if (loss_grounded == 0)
//			reward = 0;
		return reward;
	}
	
	public double buildPSLRule(int[] ruleEmbedding) throws ClassNotFoundException, IllegalAccessException, InstantiationException {
		final double initW = 5.0;
		
		FormulaContainer body = null;
		FormulaContainer head = null;
		FormulaContainer rule = null;
		/*
		 *  Add New Updated Rule
		 *  Choose Target as Head Predicate
		 */
		int posActIdx = posPredNum-1;
		int negActIdx = posPredNum+negPredNum-1;
		int headIdx;
		if (ruleEmbedding[posActIdx] == 1)
			headIdx = posActIdx;
		else
			headIdx = negActIdx;
		if (headIdx < posPredNum) { // Negative Head
			StandardPredicate p = allPosPreds[headIdx];
			List<GenericVariable> argsList = generalPredArgsMap.get(p);
			Object[] args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			head = (FormulaContainer) head.bitwiseNegate();
		} else { // Positive Head
			StandardPredicate p = allNegPreds[headIdx-posPredNum];
			List<GenericVariable> argsList = generalPredArgsMap.get(p);
			Object[] args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
		}
		
		for (int i=0; i<ReturnAction; i++) {
			if (ruleEmbedding[i]==1 && i!=headIdx) {
				if (i < posPredNum) { // Positive Body Predicate
					StandardPredicate p = allPosPreds[i];
					List<GenericVariable> argsList = generalPredArgsMap.get(p);
					Object[] args = new Object[argsList.size()];
					args = argsList.toArray(args);
					if (body==null)
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.and(f_tmp);
					}
				} else { // Negative Body Predicate
					StandardPredicate p = allNegPreds[i-posPredNum];
					List<GenericVariable> argsList = generalPredArgsMap.get(p);
					Object[] args = new Object[argsList.size()];
					args = argsList.toArray(args);
					if (body==null) {
						body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						body = (FormulaContainer) body.bitwiseNegate();
					} else {
						FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
						f_tmp = (FormulaContainer) f_tmp.bitwiseNegate();
						body = (FormulaContainer) body.and(f_tmp);
					}
				}
			}
		}
		
		if (body==null) {
			rule = head;
		} else {
			rule = (FormulaContainer) body.rightShift(head);
		}
		
		boolean succeed;
		Map<String, Object> argsMap = new HashMap<String, Object>();
		argsMap.put("rule", rule);
		argsMap.put("square", true);
		argsMap.put("weight", initW);
//		try {
//			model.add(argsMap);
//		} catch (Exception e) {
//			System.out.println("Adding Rule Fails");
//		}
		model.add(argsMap);
		// Calculate Score
		return AUCcost();
	}
	
}

