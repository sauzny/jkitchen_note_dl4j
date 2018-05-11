package com.sauzny.mnist;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import lombok.extern.slf4j.Slf4j;

/**
 * Hello world!
 *
 */
@Slf4j
public class App {

    public static void main(String[] args) throws IOException {
        
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        
        System.out.println(System.getProperty("java.io.tmpdir"));
        //number of rows and columns in the input pictures
        final int numRows = 28; // 矩阵的行数。
        final int numColumns = 28; // 矩阵的列数。
        int outputNum = 10; // 潜在结果（比如0到9的整数标签）的数量。
        
        /**
         * batchSize 和 numEpochs必须根据经验选择，而经验则需要通过实验来积累。
         * batchSize，每批次处理的数据越多，训练速度越快；
         * epoch的数量越多，遍历数据集的次数越多，准确率就越高。
         * 但是，epoch的数量达到一定的大小之后，增益会开始减少，所以要在准确率与训练速度之间进行权衡。
         * 总的来说，需要进行实验才能发现最优的数值。
         * 本示例中设定了合理的默认值。
         */
        int batchSize = 128; // 每一步抓取的样例数量。
        int numEpochs = 15; // 一个epoch指将给定数据集全部处理一遍的周期。
        
        int rngSeed = 123; // 这个随机数生成器用一个随机种子来确保训练时使用的初始权重维持一致。下文将会说明这一点的重要性。
        double rate = 0.0015; // learning rate

        
        
        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        log.info("Build model....");
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                /**
                 * 该参数将一组随机生成的权重确定为初始权重。
                 * 如果一个示例运行很多次，而每次开始时都生成一组新的随机权重，那么神经网络的表现（准确率和F1值）有可能会出现很大的差异，
                 * 因为不同的初始权重可能会将算法导向误差曲面上不同的局部极小值。
                 * 在其他条件不变的情况下，保持相同的随机权重可以使调整其他超参数所产生的效果表现得更加清晰。
                 */
            .seed(rngSeed) //include a random seed for reproducibility
            
            /**
             * 随机梯度下降（Stochastic Gradient Descent，SGD）是一种用于优化代价函数的常见方法。
             * 要了解SGD和其他帮助实现误差最小化的优化算法，可参考Andrew Ng的机器学习课程以及本网站术语表中对SGD的定义。
             */
            //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(rate, 0.98))
            .l2(rate * 0.005) // regularize learning model
            .list()
            .layer(0, new DenseLayer.Builder() //create the first input layer.
                    .nIn(numRows * numColumns)
                    .nOut(500)
                    .build())
            .layer(1, new DenseLayer.Builder() //create the second input layer
                    .nIn(500)
                    .nOut(300)
                    .build())
            .layer(2, new DenseLayer.Builder() //create the 3 input layer
                    .nIn(300)
                    .nOut(100)
                    .build())
            .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .activation(Activation.SOFTMAX)
                    .nIn(100)
                    .nOut(outputNum)
                    .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            log.info("Epoch " + i);
            model.fit(mnistTrain);
        }

        SerializableModelUtils.out(model);
        
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
    }
}

/**
指示符                         描述
Accuracy    准确率：模型准确识别出的MNIST图像数量占总数的百分比。
Precision   精确率：真正例的数量除以真正例与假正例数之和。
Recall      召回率：真正例的数量除以真正例与假负例数之和。
F1 Score    F1值  ：精确率和召回率的加权平均值。
*/

/*
 * 精确率、召回率和F1值衡量的是模型的相关性。
 * 举例来说，“癌症不会复发”这样的预测结果（即假负例/假阴性）就有风险，因为病人会不再寻求进一步治疗。
 * 所以，比较明智的做法是选择一种可以避免假负例的模型（即精确率、召回率和F1值较高），尽管总体上的准确率可能会相对较低一些。
 */
