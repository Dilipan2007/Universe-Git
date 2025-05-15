1. Define a particle with the following properties
    * Occupy some space
    * Various shapes
        * Sphere is done here
    * Rigid properties
    * Position ,velocity and angular momentum

2. Dealing with lagarngian  
     Three equations of motion for each particle, each of them decompose into system of equations to solve by RK45  
     > * For example: For a system conataining *N* particles then we have *3N* equations to solve. If the maximum order of the *3N* equations is *r*, we then have atmost _3N*r_ equations which we will give to the `solve_ivp` function of `scipy` to solve.  
     > * The _3N*r_ equations will symbolic in nature and only symbolically solved only once. With updated initial values, these are solved numerous times using a funtion inside `evolve` module, which uses `solve_ivp`.

<mark>This is a highlighted text </mark>


`code line`

```py
a=5
if a!= 10:
    print('hey')
    a+=1
```
______

[Directing to a link](www.google.com)

![An image: colliding balls](all\all\core\refs\Colliding balls\pic2.jpg)

>Attention text

|heading 1|heading 2|
|____|____|
|row 1|row2|


- [ ]hey








