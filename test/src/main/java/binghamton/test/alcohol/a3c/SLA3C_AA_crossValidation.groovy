package binghamton.test.alcohol.a3c

import binghamton.rl.A3C.generator3.a3cMDP_NN3
import binghamton.rl.A3C.pslModelCreation
import binghamton.rl.A3C.generator1_semantic.a3cMDP_NN1
import edu.umd.cs.psl.config.ConfigBundle
import edu.umd.cs.psl.config.ConfigManager
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database
import edu.umd.cs.psl.groovy.PSLModel
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.predicate.StandardPredicate


def fold = 1; //2; // 0, 1
def outputPath = 'result/cv_AA_2019/'+fold+'/'


/*
 * Training
 */
def mdp = new a3cMDP_NN1();

// Load
//mdp.loadA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser')

mdp.train();


println("Finish training")

/*
 * Save Result
 */
File outputDir = new File(outputPath);
if (!outputDir.exists()) {
	outputDir.mkdirs();
}
mdp.saveA3C(outputPath+ 'valueNet.ser', outputPath+ 'policyNet.ser');




