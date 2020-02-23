package binghamton.test.alcohol.a3c;

import binghamton.rl.A3C.generator3.a3cMDP_NN3
import binghamton.rl.A3C.pslModelCreation
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.predicate.StandardPredicate



def version = 3;

def mdp;
//if (version == 1)
////	mdp = new a3cMDP_NN();
//else if (version ==2)
//	mdp = new a3cMDP_NN2();
//else if (version == 3)
////	mdp = new a3cMDP_NN3();
if (version == 2)
	mdp = new a3cMDP_NN2();
else if (version == 3)
	mdp = new a3cMDP_NN3();

/*
 * Training
 */
def outputPath = 'result/A3C/generator'+ version+'/';
//mdp.loadA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser')
mdp.train();

println("Finish training")

//double accululatedReward = mdp.optimalSolution();
//PSLModel model = mdp.model;

/*
 * Save Result
 */
File outputDir = new File(outputPath);
if (!outputDir.exists()) {
	outputDir.mkdirs();
}

mdp.saveA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser');

//PrintStream ps = new PrintStream(new FileOutputStream(outputPath+ "result.txt", false));
//System.setOut(ps);
//System.out.println("Accumulated Reward: "+ accululatedReward);
//
//ps = new PrintStream(new FileOutputStream(outputPath+"model.txt", false));
//System.setOut(ps);
//System.out.println(""+ model.toString());






