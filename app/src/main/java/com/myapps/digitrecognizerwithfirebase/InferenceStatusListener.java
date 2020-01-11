package com.myapps.digitrecognizerwithfirebase;

public interface InferenceStatusListener {
    void onSuccess(Recognition recognition);
    void onFail(Exception e);
}
