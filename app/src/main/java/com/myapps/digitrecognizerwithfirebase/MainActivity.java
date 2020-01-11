package com.myapps.digitrecognizerwithfirebase;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.FirebaseApp;

public class MainActivity extends AppCompatActivity {

    PaintView paintView;
    TextView textView;
    Classifier classifier;
    Button checkButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        paintView = findViewById(R.id.paintView);
        textView = findViewById(R.id.textView);
        checkButton = findViewById(R.id.checkButton);
        checkButton.setEnabled(false);

        FirebaseApp.initializeApp(this);
        classifier = new Classifier(new MainModelListener(), new MainInferenceListener());
    }

    public void check(View view) {
        classifier.classify(paintView.getScaledBitmap());
    }

    public void redraw(View view) {
        paintView.redraw();
        textView.setText("Please write a digit");
    }

    private class MainModelListener implements ModelStatusListener {
        @Override
        public void onSuccess() {
            checkButton.setEnabled(true);
        }

        @Override
        public void onFail(Exception e) {
            checkButton.setEnabled(false);
            Toast.makeText(MainActivity.this, "Failed to load model", Toast.LENGTH_LONG)
                    .show();
        }
    }

    private class MainInferenceListener implements InferenceStatusListener {
        @Override
        public void onSuccess(Recognition recognition) {
            String s = "Unable to identify digit";

            if (recognition != null) {
                s = "You wrote " + recognition.getDigit() + ". I am " + (Math.round(recognition.getProb() * 100)) + "% sure.";
            }
            textView.setText(s);
        }

        @Override
        public void onFail(Exception e) {
            Toast.makeText(MainActivity.this, "Inference failed", Toast.LENGTH_LONG).show();
        }
    }
}
