package com.myapps.digitrecognizerwithfirebase;

import android.graphics.Bitmap;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Classifier {

    public static final String MODEL_NAME = "MnistModel";

    private FirebaseModelInterpreter tflite;
    private FirebaseModelInputOutputOptions inputOutputOptions;

    private final int IMAGE_SIZE = 28;
    private List<String> labels;

    private ModelStatusListener modelStatusListener;
    private InferenceStatusListener inferenceStatusListener;

    public Classifier(ModelStatusListener modelStatusListener, InferenceStatusListener inferenceStatusListener) {
        this.modelStatusListener = modelStatusListener;
        this.inferenceStatusListener = inferenceStatusListener;

        loadModelFromFirebase();

        labels = new ArrayList<>();
        for (int i = 0; i <= 9; i++) {
            labels.add(String.valueOf(i));
        }
    }

    private void loadModelFromFirebase() {
        final FirebaseCustomRemoteModel remoteModel =
                new FirebaseCustomRemoteModel.Builder(MODEL_NAME).build();
        FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions.Builder()
                .requireWifi()
                .build();

        FirebaseModelManager.getInstance().download(remoteModel, conditions)
                .addOnCompleteListener(new OnCompleteListener<Void>() {
                    @Override
                    public void onComplete(@NonNull Task<Void> task) {
                        FirebaseModelInterpreterOptions options =
                                new FirebaseModelInterpreterOptions.Builder(remoteModel).build();

                        try {
                            tflite = FirebaseModelInterpreter.getInstance(options);
                            inputOutputOptions =
                                    new FirebaseModelInputOutputOptions.Builder()
                                            .setInputFormat(0, FirebaseModelDataType.FLOAT32,
                                                    new int[]{1, 28, 28})
                                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32,
                                                    new int[]{1, 10})
                                            .build();

                            modelStatusListener.onSuccess();
                        } catch (FirebaseMLException e) {
                            e.printStackTrace();
                            modelStatusListener.onFail(e);
                        }
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        modelStatusListener.onFail(e);
                    }
                });
    }

    public void classify(final Bitmap bitmap) {
        try {
            float[][][] input = convertBitmapToFloatArray(bitmap);
            FirebaseModelInputs inputs = new FirebaseModelInputs.Builder()
                    .add(input)
                    .build();
            tflite.run(inputs, inputOutputOptions)
                    .addOnSuccessListener(new OnSuccessListener<FirebaseModelOutputs>() {
                        @Override
                        public void onSuccess(FirebaseModelOutputs result) {
                            float[][] output = result.getOutput(0);
                            float[] probabilities = output[0];

                            Map<String, Float> labeledProbabilities = new HashMap<>();
                            int i = 0;
                            for (String label : labels) {
                                labeledProbabilities.put(label, probabilities[i++]);
                            }

                            float maxProb = 0f;
                            String digit = null;
                            for (String d : labeledProbabilities.keySet()) {
                                Float p = labeledProbabilities.get(d);
                                if (p > maxProb) {
                                    maxProb = p;
                                    digit = d;
                                }
                            }
                            if (maxProb >= 0.5) {
                                Recognition recognition = new Recognition(Integer.valueOf(digit), maxProb);
                                inferenceStatusListener.onSuccess(recognition);
                            } else {
                                inferenceStatusListener.onSuccess(null);
                            }
                        }
                    })
                    .addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(@NonNull Exception e) {
                            inferenceStatusListener.onFail(e);
                        }
                    });
        } catch (FirebaseMLException e) {
            inferenceStatusListener.onFail(e);
        }
    }

    private float[][][] convertBitmapToFloatArray(Bitmap bitmap) {
        int intValues[] = new int[IMAGE_SIZE * IMAGE_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        float[][][] values = new float[1][IMAGE_SIZE][IMAGE_SIZE];
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                values[0][i][j] = intValues[i * IMAGE_SIZE + j] != -1 ? 1f : 0f;
            }
        }

        return values;
    }
}
