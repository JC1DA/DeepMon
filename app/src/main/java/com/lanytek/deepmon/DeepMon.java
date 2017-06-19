package com.lanytek.deepmon;

/**
 * Created by JC1DA on 6/19/17.
 */

public class DeepMon {
    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

    public static native void InitDeepMonWithPackageName(String package_name);
    public static native void LoadNet(String model_dir_path);
    public static native float [] GetInference(float [] input);
}
