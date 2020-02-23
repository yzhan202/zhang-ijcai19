package binghamton.rl.A3C;

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
import binghamton.util.DataOutputter;



public class pslModelCreation {
	String dataDir;
	ConfigManager cm;
	ConfigBundle config;
	DataStore data;
	PSLModel model;
	
	Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap;
	
	Database wlTruthDB;
	Partition trainPart;
	
	StandardPredicate[] X;
	StandardPredicate[] Y;
	StandardPredicate[] Z;
	StandardPredicate[] negFeatPreds;
	StandardPredicate[] friendPreds;
	StandardPredicate[] networkPreds;
	
	/*
	 * Right Reason
	 */
	Set<String> Alcohol_Signal;
	Set<String> Sober_Signal;
	
	public pslModelCreation(int threadNum) {
		dataDir = 'data/crossValidation/alcohol_data/1/training'; //'data/smallAlcohol';
		//config manager
		cm = ConfigManager.getManager();
		config = cm.getBundle("SL_alcohol"+threadNum);
		Logger log = LoggerFactory.getLogger(this.class);
		
		//database
		def defaultPath = System.getProperty("java.io.tmpdir");
		String dbpath = config.getString("dbpath", defaultPath + File.separator + "a3cSL_alcohol"+ threadNum);
		data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config);
		
		model = new PSLModel(this, data);
		
		model.add predicate: "recovers", types: [ArgumentType.UniqueID]
		model.add predicate: "attendsAA", types: [ArgumentType.UniqueID]
		model.add predicate: "localRecovers", types: [ArgumentType.UniqueID]
		model.add predicate: "usesAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.String]
		model.add predicate: "usesSoberWord", types: [ArgumentType.UniqueID, ArgumentType.String]
		model.add predicate: "containsAlcoholWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
		model.add predicate: "containsSoberWord", types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.String]
		model.add predicate: "alcoholTopic", types: [ArgumentType.UniqueID]
		model.add predicate: "soberTopic", types: [ArgumentType.UniqueID]
////		model.add predicate: "affect", types: [ArgumentType.UniqueID]
////		model.add predicate: "social", types: [ArgumentType.UniqueID]
		
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

//		def closedPredicates = [attendsAA, containsAlcoholWord, containsSoberWord, usesAlcoholWord, usesSoberWord, localRecovers,
//			alcoholTopic, soberTopic] as Set;
//		def closedPredicates = [friends, friendContainsAlcoholWord, friendContainsSoberWord, friendUsesAlcoholWord, friendUsesSoberWord,
//			replies, retweets, friendReplies, friendRetweets];
		
		def closedPredicates = [attendsAA, containsAlcoholWord, containsSoberWord, usesAlcoholWord, usesSoberWord, localRecovers,
			alcoholTopic, soberTopic, friends, friendContainsAlcoholWord, friendContainsSoberWord, 
			friendUsesAlcoholWord, friendUsesSoberWord, replies, retweets, friendReplies, friendRetweets] as Set
		def inferredPredicates = [recovers] as Set;
		
		negFeatPreds = [localRecovers, alcoholTopic, soberTopic]
		friendPreds = [friends, friendContainsAlcoholWord, friendContainsSoberWord, friendUsesAlcoholWord, friendUsesSoberWord,
			replies, retweets, friendReplies, friendRetweets];
//		networkPreds = [friends, replies, retweets, friendReplies, friendRetweets];
		
		Alcohol_Signal = ["CONTAINSALCOHOLWORD", "USESALCOHOLWORD", "~LOCALRECOVERS", "ALCOHOLTOPIC", 
			"FRIENDCONTAINSALCOHOLWORD", "FRIENDUSESALCOHOLWORD"] as Set;
		Sober_Signal = ["CONTAINSSOBERWORD", "USESSOBERWORD", "LOCALRECOVERS", "SOBERTOPIC",
			"FRIENDCONTAINSSOBERWORD", "FRIENDUSESSOBERWORD"] as Set;
		
		def predicateFileMap = [((Predicate)recovers):"recovers.txt",
			((Predicate)localRecovers):"localRecovers.txt",
			((Predicate)usesAlcoholWord):"usesAlcoholWord.txt",
			((Predicate)usesSoberWord):"usesSoberWord.txt",
			((Predicate)containsAlcoholWord):"containsAlcoholWord.txt",
			((Predicate)containsSoberWord):"containsSoberWord.txt",
			((Predicate)alcoholTopic):"alcoholTopic.txt",
			((Predicate)soberTopic):"soberTopic.txt",
			((Predicate)attendsAA):"attendsAA.txt",
////			((Predicate)affect):"affect.txt",
////			((Predicate)social):"social.txt"]
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
			((Predicate)localRecovers):true,
			((Predicate)usesAlcoholWord):false,
			((Predicate)usesSoberWord):false,
			((Predicate)containsAlcoholWord):false,
			((Predicate)containsSoberWord):false,
			((Predicate)alcoholTopic):false,
			((Predicate)soberTopic):false,
			((Predicate)attendsAA):false,
			////			((Predicate)affect):true,
			////			((Predicate)social):true,
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
		
		generalPredArgsMap = [((Predicate)recovers): [U],
			((Predicate)localRecovers): [U],
			((Predicate)usesAlcoholWord): [U,AW],
			((Predicate)usesSoberWord): [U,SW],
			((Predicate)containsAlcoholWord): [U,I,AW],
			((Predicate)containsSoberWord): [U,I,SW],
//			((Predicate)affect): [U],
//			((Predicate)social): [U],
			((Predicate)alcoholTopic): [U],
			((Predicate)soberTopic): [U],
			((Predicate)attendsAA): [U],
			
			((Predicate)friends): [U,U1],
			((Predicate)friendContainsAlcoholWord): [U1,I1,AW1],
			((Predicate)friendContainsSoberWord): [U1,I1,SW1],
			((Predicate)friendUsesAlcoholWord): [U1,AW1],
			((Predicate)friendUsesSoberWord): [U1,SW1],
			((Predicate)replies): [U,U1,I],
			((Predicate)retweets): [U,U1,I],
			((Predicate)friendReplies): [U1,U,I1],
			((Predicate)friendRetweets): [U1,U,I1]]
		
		
		trainPart = new Partition(0);
		Partition truthPart = new Partition(1)
		
		def inserter;
		for (Predicate p: closedPredicates) {
			String fileName = predicateFileMap[p];
			inserter = data.getInserter(p, trainPart);
			def fullFilePath = dataDir + '/' + fileName;
//			println p.toString()
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

		wlTruthDB = data.getDatabase(truthPart, inferredPredicates);
		
		X = closedPredicates.toArray();
		Y = inferredPredicates.toArray();
		Z = [];
	}
	
	private List<String> getAlcoholWords(){
		def fullFilePath = "data/topAlcoholWords.txt";
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
	
	
	void populateDatabase(DataStore data, Database dbToPopulate, Partition populatePartition, Set inferredPredicates){
		Database populationDatabase = data.getDatabase(populatePartition, inferredPredicates);
		DatabasePopulator dbPop = new DatabasePopulator(dbToPopulate);
	
		for (Predicate p : inferredPredicates){
			dbPop.populateFromDB(populationDatabase, p);
		}
		populationDatabase.close();
	}
} 












