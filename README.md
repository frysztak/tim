## Tim
Tim is my humble attempt to dive into the world of computer vision. 
So far, it can:
* substract background (SIMD-optimized Grimson-Stauffer algorithm, with improved foreground detection)
* remove shadows
* detect and count moving objects

Keep in mind it's still work in progress.

### Installation
```
git clone https://github.com/sebastian-frysztak/tim.git
cd tim
mkdir build
cd build
cmake ..
make
```

### Usage
In `build` directory run `./tim lausanne`, you should see a pretty self-explanatory window.

Tim keeps configuration in JSON files, in `data` dir. Video files need to have the same name as JSON file, but `.mp4` extension.
While Tim is running you can run scripts from `scripts` folder.

You can also run benchmark mode by adding `--b` to arguments.

### CMake options
Probably most noteworthy option is `SIMD`. It enables SIMD-optimized (so far only SSE2 is implemented) background substraction code. On Intel i7-2640M it runs about 2.5 times faster than scalar code. It's enabled by default.

If you want deeper understanding how shadow removal works, you can use `DEBUG`. Keep in mind that for Lausanne video shadow removal is disabled (as there's no need to remove shadows).

### Used publications
1. Chris Stauffer, W.E.L Grimson, "Adaptive background mixture models for real-time tracking"
2. Csaba Benedek, Tamás Szirányi, "Bayesian Foreground and Shadow Detection in Uncertain Frame Rate Surveillance Videos" 
3. Ariel Amato,  Mikhail G. Mozerov,  Andrew D. Bagdanov, and  Jordi Gonzàlez, "Accurate Moving Cast Shadow Suppression Based on Local Color Constancy Detection"
