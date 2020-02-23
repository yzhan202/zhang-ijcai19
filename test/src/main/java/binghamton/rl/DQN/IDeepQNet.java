package binghamton.rl.DQN;

import java.io.IOException;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import binghamton.rl.NeuralNet;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 *
 * This neural net quantify the value of each action given a state
 *
 */
public interface IDeepQNet<NN extends IDeepQNet> extends NeuralNet<NN> {

    boolean isRecurrent();

    void reset();

    void fit(INDArray input, INDArray labels);

    void fit(INDArray input, INDArray[] labels);

    INDArray output(INDArray batch);

    INDArray[] outputAll(INDArray batch);

    NN clone();

    void copy(NN from);

    Gradient[] gradient(INDArray input, INDArray label);

    Gradient[] gradient(INDArray input, INDArray[] label);

    void applyGradient(Gradient[] gradient, int batchSize);

    double getLatestScore();
    
}
