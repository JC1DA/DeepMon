package com.lanytek.deepmon;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.squareup.picasso.Picasso;

import java.io.File;
import java.io.IOException;

import static com.lanytek.deepmon.Utilities.convert_yolo_detections;
import static com.lanytek.deepmon.Utilities.do_nms_sort;
import static com.lanytek.deepmon.Utilities.getColorPixel;

public class MainActivity extends AppCompatActivity {
    public static final String TAG = "DEEPMON";

    private static final String [] yolo_descriptions = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    public static final int MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 9999;

    private static final int SELECT_PICTURE = 9999;
    private String selectedImagePath = null;

    // Used to load the 'native-lib' library on application startup.

    private Activity activity = this;
    private Button btn_Init;
    private Button btn_Read_Net;
    private Button btn_Process;

    ImageView iv;

    static {
        System.loadLibrary("deepmon");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn_Init = (Button) findViewById(R.id.btn_Init);
        btn_Read_Net = (Button) findViewById(R.id.btn_read_net);
        btn_Process = (Button) findViewById(R.id.btn_process);

        iv = (ImageView) findViewById(R.id.iv_image);

        iv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_PICTURE);
            }
        });

        btn_Init.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Utilities.copyFile(activity, "common.cl");
                Utilities.copyFile(activity, "im2col.cl");
                Utilities.copyFile(activity, "conv.cl");
                Utilities.copyFile(activity, "pooling.cl");
                Utilities.copyFile(activity, "fc.cl");
                Utilities.copyFile(activity, "activation.cl");
                InitDeepMonWithPackageName(activity.getPackageName().toString());
            }
        });

        btn_Read_Net.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/Yolo-Tiny";
                LoadNet(path);
            }
        });

        btn_Process.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new async_processImage_yolo().execute();
                //TestInference();
            }
        });

        if (ContextCompat.checkSelfPermission(activity,
                Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED) {

            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(activity,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)) {

                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.

            } else {

                // No explanation needed, we can request the permission.

                ActivityCompat.requestPermissions(activity,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);

                // MY_PERMISSIONS_REQUEST_READ_CONTACTS is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        }
    }

    private class async_processImage_yolo extends AsyncTask<Void, Void, Void> {

        private double t1,t2;
        private double cnn_runtime;
        private float [] result;
        private Bitmap bm = null;

        @Override
        protected void onPreExecute() {
            btn_Process.setEnabled(false);
            t1 = System.currentTimeMillis();
            super.onPreExecute();
        }

        @Override
        protected Void doInBackground(Void... params) {
            if(selectedImagePath != null) {
                final int IMG_X = 448;
                final int IMG_Y = 448;
                final int IMG_C = 3;

                final float [] bitmapArray = new float[IMG_X * IMG_Y * IMG_C];

                try {
                    bm = Picasso.with(activity)
                            .load(new File(selectedImagePath))
                            .config(Bitmap.Config.ARGB_8888)
                            .resize(IMG_X,IMG_Y)
                            .get();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if(bm != null) {


                    for(int w = 0 ; w < bm.getWidth() ; w++) {
                        for(int h = 0 ; h < bm.getHeight() ; h++) {
                            int pixel = bm.getPixel(w, h);
                            for(int c = 0 ; c < 3 ; c++) {
                                bitmapArray[h * IMG_X * IMG_C + w * IMG_C + c] = getColorPixel(pixel, c);
                            }
                        }
                    }
                }

                double x1 = System.currentTimeMillis();
                float [] result = GetInference(bitmapArray);
                double x2 = System.currentTimeMillis();
                cnn_runtime = x2 - x1;
                Log.d(TAG,"CNN RUNTIME: " + cnn_runtime + "ms");

                int classes = 20;
                int side = 7;
                int num = 2;
                float thresh = 0.15f;

                //process result first
                float [][] probs = new float[side * side * num][classes];
                Utilities.box[] boxes = new Utilities.box[side * side * num];
                for(int j = 0 ; j < boxes.length ; j++)
                    boxes[j] = new Utilities.box();

                convert_yolo_detections(result, classes, num, 1, side, 1, 1, thresh, probs, boxes, 0);

                do_nms_sort(boxes, probs, side * side * num, classes, 0.5f);

                //do box drawing
                final Bitmap mutableBitmap = Bitmap.createScaledBitmap(
                        bm, 512, 512, false).copy(bm.getConfig(), true);
                final Canvas canvas = new Canvas(mutableBitmap);

                for(int i = 0; i < side * side * num; ++i){

                    int classid = -1;
                    float maxprob = -100000.0f;
                    for(int j = 0 ; j < classes ; j++) {
                        if(probs[i][j] > maxprob) {
                            classid = j;
                            maxprob = probs[i][j];
                        }
                    }

                    if(classid < 0)
                        continue;

                    float prob = probs[i][classid];
                    if(prob > thresh){
                        /*int width = 8;
                        //classid = classid * side % classes;
                        float red = get_color(0,classid,classes);
                        float green = get_color(1,classid,classes);
                        float blue = get_color(2,classid,classes);*/
                        Utilities.box b = boxes[i];

                        int left  = (int) ((b.x-b.w/2.) * mutableBitmap.getWidth());
                        int right = (int) ((b.x+b.w/2.) * mutableBitmap.getWidth());
                        int top   = (int) ((b.y-b.h/2.) * mutableBitmap.getHeight());
                        int bot   = (int) ((b.y+b.h/2.) * mutableBitmap.getHeight());

                        if(left < 0) left = 0;
                        if(right > mutableBitmap.getWidth() - 1) right = mutableBitmap.getWidth() - 1;
                        if(top < 0) top = 0;
                        if(bot > mutableBitmap.getHeight() - 1) bot = mutableBitmap.getHeight() - 1;

                        Paint p = new Paint();
                        p.setStrokeWidth(p.getStrokeWidth() * 3);
                        p.setColor(Color.RED);
                        canvas.drawLine(left, top, right, top, p);
                        canvas.drawLine(left, top, left, bot, p);
                        canvas.drawLine(left, bot, right, bot, p);
                        canvas.drawLine(right, top, right, bot, p);

                        p.setTextSize(48f);
                        p.setColor(Color.BLUE);
                        canvas.drawText("" + yolo_descriptions[classid],left + (right - left)/2,top + (bot - top)/2,p);
                    }
                }

                activity.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        iv.setImageBitmap(mutableBitmap);
                    }
                });
            }

            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            t2 = System.currentTimeMillis();
            double runtime = t2 - t1;
            btn_Process.setEnabled(true);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                    // permission was granted, yay! Do the
                    // contacts-related task you need to do.

                } else {

                    // permission denied, boo! Disable the
                    // functionality that depends on this permission.
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    public String getPath(Uri uri) {
        String[] projection = { MediaStore.Images.Media.DATA };
        Cursor cursor = managedQuery(uri, projection, null, null, null);
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        return cursor.getString(column_index);
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData();
                selectedImagePath = getPath(selectedImageUri);
                if(selectedImagePath != null)
                    iv.setImageURI(selectedImageUri);
            }
        }
    }


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native void testDeepMon();
    public native void InitDeepMonWithPackageName(String package_name);
    public native void LoadNet(String model_dir_path);
    public native float [] GetInference(float [] input);
    public native float [] TestInference();
}
