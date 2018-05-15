package com.sauzny.mnist;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

public final class TrainModelUtils {

    private TrainModelUtils(){}

    static int height = 28;
    static int width = 28;
    static int channels = 1;
    static int rngseed = 123;
    static Random randNumGen = new Random(rngseed);
    static int batchSize = 128;
    static int outputNum = 10;
    static int numEpochs = 1;
    
    public static DataSetIterator createDataSetIterator(String dirPath) throws IOException{

        File trainData = new File(dirPath);
        
        // Define the FileSplit(PATH, ALLOWED FORMATS,random)
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        
        // Extract the parent path as the image label
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name
        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        
        return dataIter;
    }
    
    public static DataSetIterator mnistTrain(){
        try {
            return createDataSetIterator("E:\\data01\\dl4j\\mnist_png\\training");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return null;
    }
    
    public static DataSetIterator mnistTest(){
        try {
            return createDataSetIterator("E:\\data01\\dl4j\\mnist_png\\testing");
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return null;
    }
    
    public static void main(String[] args) {
        
    }
}
