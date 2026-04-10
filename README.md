
# Instructions for Running Code on Clusters

## Individual Implementation Instructions

1. Navigate to ondemand.chpc.utah.edu

    - You can also use SSH to log in directly from your  terminal, Example:
      ```bash
      ssh username@kingspeak1.chpc.utah.edu
      ```

2. Log into your account and do the 2 factor authentication.

3. In the top left corner cick ont he `clusters` drop down menu and select 'Notchpeak Shell Access'

    <img src = pictures/image.png>

4. At this point in the process you have 2 options
    1. Manually copy project files to the cluster
    2. Use git to clone the repository

5. Clone the repository to your home directory:

    ```bash
    git clone git@github.com:APkenobi175/GenreRevealParty.git
    ```

6. Download dataset from [here](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs?resource=download)

    - On the dashboard of ondemand.chpc.utah.edu, click on the `files` tab in the top left menu, and select `Home Directory`

    - Navigate to `GenreRevealParty` directory

    - Click on the `New Directory` button and create a new directory called `data`

      <img src = pictures/image2.png>

    - Unzip, and move `tracks_features.csv` to the `GenreRevealParty/data` directory.

7. Back in the shell, Load the module for your desired implementation (e.g., OpenMP, CUDA, OpenMPI):
    - Examples:
      - For CUDA:
            `module load cuda`
      - For OpenMPI:
            `module load openmpi`

8. Request GPU nodes, and task allocation for running the program. Example for requesting 4 tasks and 2 GPUs:

    ```bash
    srun --ntasks=4 --account=notchpeak-gpu --partition=notchpeak-gpu --gres=gpu:2 --pty bash
    ```

9. Navigate to the implementation directory and build the program using CMake

    ```bash
    cd GenreRevealParty/parallel_cuda
    ```

    - Compile the program using CMake and then return to the implementation directory:

        ```bash
        mkdir build
        cd build
        cmake ..
        make
        cd ..
        ```

10. Run the program

    - For non MPI implementations, you can run the program directly from the implementation directory using the following command:

        ```bash
        ./build/GenreRevealParty<IMPLEMENTATIONTYPE>
        ```

    - If you are using an MPI implemenation you can run the following commands from the implementation directory:

        - OpenMPI

        ```bash
        mpirun -np <NUMBER_OF_PROCESSES> ./build/GenreRevealPartyMPI
        ```

        - Cuda With MPI

        ```bash
        mpirun -np <NUMBER_OF_PROCESSES> ./build/mpi-cuda
        ```

## Running All Implementations At Once

To run all implementations at once you can run the included sbatch script `runall.sh`. This will automatically request resources, and run all implementations sequentially. To run the script use the following command from any directory on notchpeak.

```bash
sbatch runall.sh
```

## Important Notes

1. Always make sure to request a GPU node when running GPU implementations, otherwise your code will not run properly and may cause errors.

2. always run the code from the implementation directory (e.g., `GenreRevealParty/parallel_cuda`) to ensure that the correct paths are used for data and other resources. E.G do not run from the build directory, as this may cause issues with finding the data file or other resources.

3. Dataset is not stored in the repo and must be downlaoded separately due to its size. Make sure to place the `tracks_features.csv` file in the `GenreRevealParty/data` directory for the code to run properly.
