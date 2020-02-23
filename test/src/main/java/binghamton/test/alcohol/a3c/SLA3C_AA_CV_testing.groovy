package binghamton.test.alcohol.a3c

import java.text.DecimalFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import edu.umd.cs.psl.application.inference.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.config.*;
import edu.umd.cs.psl.core.*;
import edu.umd.cs.psl.core.inference.*;
import edu.umd.cs.psl.database.*;
import edu.umd.cs.psl.database.rdbms.*;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import edu.umd.cs.psl.evaluation.result.*;
import edu.umd.cs.psl.evaluation.statistics.*;
import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.atom.*;
import edu.umd.cs.psl.model.formula.*;
import edu.umd.cs.psl.model.function.*;
import edu.umd.cs.psl.model.kernel.*;
import edu.umd.cs.psl.model.kernel.rule.AbstractRuleKernel
import edu.umd.cs.psl.model.kernel.rule.CompatibilityRuleKernel
import edu.umd.cs.psl.model.predicate.*;
import edu.umd.cs.psl.model.term.*;
import edu.umd.cs.psl.model.rule.*;
import edu.umd.cs.psl.model.weight.*;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Term
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*;
import edu.umd.cs.psl.util.database.*;
import com.google.common.collect.Iterables;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.evaluation.resultui.printer.*;
import java.io.*;
import java.util.*;
import groovy.time.*;

import binghamton.util.FoldUtils
import binghamton.util.GroundingWrapper
import binghamton.rl.DQN.generator1.MDP_NN
import binghamton.rl.DQN.generator1.dqnMDP_NN
import binghamton.rl.DQN.generator2.dqnMDP_NN2
import binghamton.rl.DQN.hindsight.dqnMDP_hindsight
import binghamton.util.DataOutputter;

import org.apache.commons.lang3.ArrayUtils



def fold = 4;
String dataDir = 'data/crossValidation/alcohol_data/'+ fold+ '/test'

//config manager
ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("SL_alcohol")
Logger log = LoggerFactory.getLogger(this.class)

//database
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "structureLearning_alcohol")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

PSLModel model = new PSLModel(this, data)

List<String> getAlcoholWords(){
	def fullFilePath = "data/alcohol_test/topAlcoholWords.txt";
	BufferedReader br = new BufferedReader(new FileReader(fullFilePath));
	List<String> topAlcoholWords = new ArrayList<String>();

	try {
		String line = br.readLine();
		while (line != null) {
			topAlcoholWords.add(line)
			line = br.readLine();
		}
	} finally {
		br.close();
	}
	return topAlcoholWords;
}

model.add predicate: "recovers", types: [ArgumentType.UniqueID]
model.add predicate: "attendsAA", types: [ArgumentType.UniqueID]
model.add predicate: "localRecovers", types: [ArgumentType.UniqueID]
model.add predicate: "usesAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "usesSoberWord", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "containsAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "containsSoberWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "alcoholTopic", types: [ArgumentType.UniqueID]
model.add predicate: "soberTopic", types: [ArgumentType.UniqueID]

// Friend Feature
model.add predicate: "friends", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "friendContainsAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "friendContainsSoberWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "friendUsesAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.String]
model.add predicate: "friendUsesSoberWord", types: [ArgumentType.UniqueID, ArgumentType.String]

model.add predicate: "replies", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "retweets", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "friendReplies", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
model.add predicate: "friendRetweets", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]


def closedPredicates = [attendsAA, containsAlcoholWord, containsSoberWord, localRecovers, usesAlcoholWord, usesSoberWord,
	alcoholTopic, soberTopic, friends, friendContainsAlcoholWord, friendContainsSoberWord, friendUsesAlcoholWord, friendUsesSoberWord, 
	replies, retweets, friendReplies, friendRetweets] as Set;

def inferredPredicates = [recovers] as Set;

StandardPredicate[] negFeatPreds = [localRecovers, alcoholTopic, soberTopic]
StandardPredicate[] friendPreds = [friends, friendContainsAlcoholWord, friendContainsSoberWord, friendUsesAlcoholWord, friendUsesSoberWord,
	replies, retweets, friendReplies, friendRetweets];
StandardPredicate[] networkPreds = [friends, replies, retweets, friendReplies, friendRetweets];

def predicateFileMap = [((Predicate)recovers):"recovers.txt",
	((Predicate)attendsAA):"attendsAA.txt",
	((Predicate)localRecovers):"localRecovers.txt",
	((Predicate)usesAlcoholWord):"usesAlcoholWord.txt",
	((Predicate)usesSoberWord):"usesSoberWord.txt",
	((Predicate)containsAlcoholWord):"containsAlcoholWord.txt",
	((Predicate)containsSoberWord):"containsSoberWord.txt",
	((Predicate)alcoholTopic):"alcoholTopic.txt",
	((Predicate)soberTopic):"soberTopic.txt",
	((Predicate)friends):"friends.txt",
	((Predicate)friendContainsAlcoholWord):"friendContainsAlcoholWord.txt",
	((Predicate)friendContainsSoberWord):"friendContainsSoberWord.txt",
	((Predicate)friendUsesAlcoholWord):"friendUsesAlcoholWord.txt",
	((Predicate)friendUsesSoberWord):"friendUsesSoberWord.txt",
	((Predicate)replies):"replies.txt",
	((Predicate)retweets):"retweets.txt",
	((Predicate)friendReplies):"friendReplies.txt",
	((Predicate)friendRetweets):"friendRetweets.txt"]


def predicateSoftTruthMap = [((Predicate)recovers):true,
	((Predicate)attendsAA):false,
	((Predicate)localRecovers):true,
	((Predicate)usesAlcoholWord):false,
	((Predicate)usesSoberWord):false,
	((Predicate)containsAlcoholWord):false,
	((Predicate)containsSoberWord):false,
	((Predicate)alcoholTopic):false,
	((Predicate)soberTopic):false,
	((Predicate)friends):false,
	((Predicate)friendContainsAlcoholWord):false,
	((Predicate)friendContainsSoberWord):false,
	((Predicate)friendUsesAlcoholWord):false,
	((Predicate)friendUsesSoberWord):false,
	((Predicate)replies):false,
	((Predicate)retweets):false,
	((Predicate)friendReplies):false,
	((Predicate)friendRetweets):false]


GenericVariable U = new GenericVariable('U', model);
GenericVariable AW = new GenericVariable('AW', model);
GenericVariable SW = new GenericVariable('SW', model);
GenericVariable I = new GenericVariable('I', model);
// Arguments for Friends
GenericVariable U1 = new GenericVariable('U1', model);
GenericVariable I1 = new GenericVariable('I1', model);
GenericVariable AW1 = new GenericVariable('AW1', model);
GenericVariable SW1 = new GenericVariable('SW1', model);

Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap = [((Predicate)recovers): [U],
	((Predicate)attendsAA): [U], 
	((Predicate)localRecovers): [U],
	((Predicate)usesAlcoholWord): [U,AW],
	((Predicate)usesSoberWord): [U,SW],
	((Predicate)containsAlcoholWord): [U,I,AW],
	((Predicate)containsSoberWord): [U,I,SW],
	((Predicate)alcoholTopic): [U],
	((Predicate)soberTopic): [U],
	((Predicate)friends): [U,U1],
	((Predicate)friendContainsAlcoholWord): [U1,I1,AW1],
	((Predicate)friendContainsSoberWord): [U1,I1,SW1],
	((Predicate)friendUsesAlcoholWord): [U1,AW1],
	((Predicate)friendUsesSoberWord): [U1,SW1],
	((Predicate)replies): [U,U1,I],
	((Predicate)retweets): [U,U1,I],
	((Predicate)friendReplies): [U1,U,I1],
	((Predicate)friendRetweets): [U1,U,I1]]

//def topAlcoholWords = getAlcoholWords();
//def topSoberWords = ['#recovery', 'sober', 'sobriety', 'recovery', '#sobriety'];


/*
 * Parse PSL Rules from Text File
 */
boolean isPredicateString(String str) {
	if (str.contains(")") || str.contains(",") || str=="")
		return false;
	else
		return true;
}
def modelSize = 0;
String modelFile = 'result/MLNs/aa/'+fold+'/l1Model_04292019.txt' //'result/crossValidation_AA/'+fold+'/model.txt'
BufferedReader reader = new BufferedReader(new FileReader(modelFile));
String read = null;
while ((read = reader.readLine()) != null) {	
	FormulaContainer body = null;
	FormulaContainer head = null;
	FormulaContainer rule = null;
	
	boolean LOGIC_NOT = false;
	String[] splited = read.split(">>");
	String bodyPart = splited[0];
	String headPart = splited[1];
	String[] head_splited = (((headPart.split("\\{"))[0]).replace(" ", "")).split("\\(");
	for (int i=0; i<head_splited.length; i++) {
		if (head_splited[i].contains("~")) {
			LOGIC_NOT = true;
		} else if (isPredicateString(head_splited[i])) {
			StandardPredicate p = (StandardPredicate)PredicateFactory.getFactory().getPredicate(head_splited[i]); // 	
			List<GenericVariable> argsList = generalPredArgsMap.get(p);
			Object[] args = new Object[argsList.size()];
			args = argsList.toArray(args);
			head = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
			
			if (LOGIC_NOT) {
				head = (FormulaContainer) head.bitwiseNegate();
				LOGIC_NOT = false;
			}
		}
	}
	
	String weightPart = bodyPart.split("\\}")[0];
	bodyPart = bodyPart.split("\\}")[1].replace(" ", "");
	double weightValue = Double.parseDouble(weightPart.replace("{", ""));
	
	if (weightValue < 3.0)
		continue
	
	String[] body_splited = bodyPart.split("&");
	for (int i=0; i<body_splited.length; i++) {
		String[] tmp = body_splited[i].split("\\(");
		for (int j=0; j<tmp.length; j++) {
			if (tmp[j].contains("~")) {
				LOGIC_NOT = true;
			} else if (isPredicateString(tmp[j])) {
				StandardPredicate p = (StandardPredicate)PredicateFactory.getFactory().getPredicate(tmp[j]);	
				List<GenericVariable> argsList = generalPredArgsMap.get(p);
				Object[] args = new Object[argsList.size()];
				args = argsList.toArray(args);
				
				if (body == null) {
					body = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					if (LOGIC_NOT) {
						body = (FormulaContainer) body.bitwiseNegate();
						LOGIC_NOT = false;
					}
				} else {
					FormulaContainer f_tmp = (FormulaContainer) model.createFormulaContainer(p.getName(), args);
					if (LOGIC_NOT) {
						f_tmp = (FormulaContainer) f_tmp.bitwiseNegate();
						LOGIC_NOT = false;
					}
					body = (FormulaContainer) body.and(f_tmp);
				}
			}
		}
	}
	
	rule = (FormulaContainer) body.rightShift(head);
	Map<String, Object> argsMap = new HashMap<String, Object>();
	argsMap.put("rule", rule);
	argsMap.put("sqaured", true);
	argsMap.put("weight", weightValue);
	
	model.add(argsMap);
	modelSize++;
	
	if (modelSize > 25)
		break;
}
reader.close();

System.out.println(model.toString());

Partition trainPart = new Partition(0)
Partition truthPart = new Partition(1)
Partition inferenceWritePart = new Partition(2);

def inserter;
for (Predicate p: closedPredicates) {
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, trainPart);
	def fullFilePath = dataDir + '/' + fileName;
	if (predicateSoftTruthMap[p]) {
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	} else {
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

for (Predicate p: [recovers]) {
	String fileName = predicateFileMap[p];
	inserter = data.getInserter(p, truthPart);
	def fullFilePath = dataDir + '/' + fileName;
	if(predicateSoftTruthMap[p]){
		InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
	}
	else{
		InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
	}
}

void populateDatabase(DataStore data, Database dbToPopulate, Partition populatePartition, Set inferredPredicates){
	Database populationDatabase = data.getDatabase(populatePartition, inferredPredicates);
	DatabasePopulator dbPop = new DatabasePopulator(dbToPopulate);

	for (Predicate p : inferredPredicates){
		dbPop.populateFromDB(populationDatabase, p);
	}
	populationDatabase.close();
}

Database inferenceDB = data.getDatabase(inferenceWritePart, closedPredicates, trainPart);
populateDatabase(data, inferenceDB, truthPart, inferredPredicates);

/*
 * Inference
 */
println("Doing Inference")
MPEInference mpe = new MPEInference(model, inferenceDB, config)
FullInferenceResult result = mpe.mpeInference()
inferenceDB.close();

Database resultsDB = data.getDatabase(inferenceWritePart, inferredPredicates)

def comparator = new SimpleRankingComparator(resultsDB)
def groundTruthDB = data.getDatabase(truthPart, inferredPredicates)
def list = Queries.getAllAtoms(groundTruthDB, recovers);

comparator.setBaseline(groundTruthDB)

def metrics = [RankingScore.AUPRC, RankingScore.NegAUPRC, RankingScore.AreaROC]
double [] score = new double[metrics.size()]
for (int i = 0; i < metrics.size(); i++) {
	comparator.setRankingScore(metrics.get(i))
	score[i] = comparator.compare(recovers)
}

System.out.println("Area under positive-class PR curve: " + score[0])
System.out.println("Area under negative-class PR curve: " + score[1])
System.out.println("Area under ROC curve: " + score[2])




