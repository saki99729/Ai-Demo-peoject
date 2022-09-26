package com.example.testprojectai;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.testprojectai.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
ImageView imageView;
TextView result,confidence;
Button picture;
int  imageSize=224;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView= findViewById(R.id.imageView);
        picture =findViewById(R.id.button);
        result =findViewById(R.id.result);
        confidence=findViewById(R.id.confidence);


        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode==1 && resultCode==RESULT_OK)
        {
            Bitmap image =(Bitmap) data.getExtras().get("data");
            int dimension =Math.min(image.getWidth(),image.getHeight());
            image= ThumbnailUtils.extractThumbnail(image,dimension,dimension);
            imageView.setImageBitmap(image);
            image= Bitmap.createScaledBitmap(image,imageSize,imageSize,false);
            ClassifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void ClassifyImage(Bitmap image) {
        try {
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            int[] intvalues = new int[imageSize * imageSize];
            image.getPixels(intvalues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++)
            {
                for(int j=0;j<imageSize;j++)
                {
                    int val=intvalues[pixel++];
                    byteBuffer.putFloat(((val >>16) & 0XFF)*(1.f/255.f));
                    byteBuffer.putFloat(((val >>8) & 0XFF)*(1.f/255.f));
                    byteBuffer.putFloat((val & 0XFF) *(1.f/255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences =outputFeature0.getFloatArray();
            int maxPos=0;
            float maxConfidence=0;

            for(int i=0; i< confidences.length;i++)
            {
                if(confidences[i]>maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[]classes ={"Apple","Avacado"};

            result.setText(classes[maxPos]);

            String s ="";
            for(int i=0; i<classes.length;i++)
            {
                s +=String.format("%s: %1.f%%\n",classes[i],confidences[i]*100);
            }
            confidence.setText(s);
            // Releases model resources if no longer used
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }



    }
}