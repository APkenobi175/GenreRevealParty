
# Instructions for Running Code on Clusters

1. Connect to cluster using SSH:

   ```bash
   ssh username@kingspeak1.chpc.utah.edu
   ```

2. Log into your account and do the 2 factor authentication.

3. Clone the repository to your home directory:

    ```bash
    git clone git@github.com:APkenobi175/GenreRevealParty.git
    ```

4. Download dataset from [here](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?resource=download)
    - Unzip, and move `tracks_features.csv` to the `GenreRevealParty/data` directory.

5. Load the module for your desired implementation (e.g., OpenMP, CUDA, OpenMPI):
    - Examples:
      - For OpenMP:
            `module load openmp`
      - For CUDA:
            `module load cuda`
      - For OpenMPI:
            `module load openmpi`

6. If running on GPU, request a GPU node:

    ```bash
    srun --account=kingspeak-gpu --partition=kingspeak-gpu --gres=gpu:1 --pty bash
    ```

7. Compile the code using the provided Makefiles, in the appropriate directory (e.g., `GenreRevealParty/parallel_cuda/build` for CUDA implementation):

    ```bash
    cd -p GenreRevealParty/<IMPLEMENTATION_DIRECTORY>/build
    cmake ..
    make
    ```

8. Return to the implementation directory and run the program

    ```bash
    cd ..
    ./build/GenreRevealParty<IMPLEMENTATIONTYPE>>
    ```

    - Example for CUDA:

        ```bash
        cd -p GenreRevealParty/parallel_cuda/build
        cmake ..
        make
        cd ..
        ./build/GenreRevealPartyCUDA
        ```

    > note: use the -p flag with cd because the build directory is not in the repository and is included in the .gitignore file

## Important Notes

1. Always make sure to request a GPU node when running GPU implementations, otherwise your code will not run properly and may cause errors.

2. always run the code from the implementation directory (e.g., `GenreRevealParty/parallel_cuda`) to ensure that the correct paths are used for data and other resources. E.G do not run from the build directory, as this may cause issues with finding the data file or other resources.

3. Dataset is not stored in the repo and must be downlaoded separately due to its size. Make sure to place the `tracks_features.csv` file in the `GenreRevealParty/data` directory for the code to run properly.


