# Second delivery for the ADM project
#### Jan Herlyn & Adrià Lisa

Under the folder 'scripts' you will find all the codes for the NeuralODE and Lokta-Volterra models, written in Julia. The codes used for the Linear Time Series models, like AR, SARIMA, etc. are all combined in the `ipynb` notebook. We provide an explanation on how to run the julia scripts, as it is not as well known as the python notebook.

### Installing Julia
To install Julia, you can go the [official website](https://julialang.org/downloads/) and follow the appropriate steps for your system. 

If you have added Julia to PATH, you can start a new session into the Julia REPL by just running "julia" (or else, run "/path/to/julia/bin/julia)" on your favorite command interface.
```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.1 (2022-09-06)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```
Feel free to play arround with the REPL to be more familiarized with the language. We encourage checking out the [offical documentation manual](https://docs.julialang.org/en/v1/manual/getting-started/), and this [cool tutorial](https://youtu.be/EkgCENBFrAY?si=DTJ3SP1Shm0wYKTk) by Miguel Raz.

### Instantiating the Virtual Environment
Our project requires a set of very specific dependencies, which are put together in a virtual environment named `p2env`. To activate the environment, open the julia RELP within this project folder, and type the character `]` to open the package manager interface. From there, you can activate it, and then you wil have to instantiate it to download all the packages. 

```
julia> ]
(@v1.10) pkg> activate p2env
(p2env) pkg> instantiate
```
Don't be surprised if the precompilation takes several minutes. You may want to go and get a coffee in the meantime ☕.

### Running the scripts

After that, you can press 'return' to go back to the normal interface. Now you are ready to run the scripts via the `include` command.
```
julia> include("scripts/PureLV.jl")
julia> include("scripts/NODE_hidden5.jl")
julia> include("scripts/NODE_hidden4.jl")
julia> include("scripts/HybridNODE.jl")
```
And that's it! While every script runs you should be getting nice animations on how the training evolves. In the end, you will get the MSE results printed in the REPL, and a final figure with the resulting trajectories.