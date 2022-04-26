import numpy as np
from scipy import linalg
from scipy.linalg import solve_banded
from matplotlib import pyplot as plt
from ipyparallel import Client


def initialize():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as spar on
    all engines. Return the DirectView.
    """
    #set up client
    client = Client()
    #group all clients as direct view
    dview = client[ : ]
    #returnt the direction view
    return dview

def FiniteElement( grid, alpha, beta, eps ):
    '''
    :param grid: ( np.ndarray ) the parameterized grid
    :param alpha: ( float ) the minimum y value
    :param beta: ( float ) the maximum y value
    :param eps: ( float ) parameter value for the diff equation
    :return: ( np.ndarray ) the coefficients for the basiss functions
    '''
    #helper function to construct A matrix
    def a( i, j ):
        '''
        :param i: ( int ) basis function index
        :param j: ( int ) basis function index
        :return: ( float ) the integral value
        '''
        if j == i+1:
            hp1 = grid[ i + 1 ] - grid[ i ]
            return eps / hp1 + 1/2.
        elif j == i:
            h1 = grid[ i ] - grid[ i-1 ]
            hp1 = grid[ i+1 ] - grid[ i ]
            return -eps/ h1 -eps/hp1
        elif j == i - 1:
            h1 = grid[ i ] - grid[ i - 1 ]
            return eps / h1 - 1 / 2.
        else:
            return 0
    #helper funtion to construct Phi vector
    def l( j ):
        '''
        :param j: ( int ) basis function index
        :return: ( float ) integral value
        '''
        h1 = grid[ j ] - grid[ j - 1 ]
        hp1 = grid[ j + 1 ] - grid[ j ]
        return -1*( h1 + hp1 ) / 2

    #construct A
    main_diag = np.ones_like( grid )
    main_diag[ 1:-1 ] = np.array([ a( i, i )
                                   for i in range( 1, grid.size - 1 ) ])
    #A_diag = np.diag(main_diag, k=0)

    sup_diag = np.zeros( grid.size )
    sup_diag[ 2: ] = np.array([ a( i + 1, i )
                                for i in range( 1, grid.size - 1 ) ])
    #A_sup_diag = np.diag(sup_diag, k=1)

    sub_diag = np.zeros( grid.size )
    sub_diag[ :-2 ] = np.array([ a( i, i + 1 )
                                 for i in range( 0, grid.size - 2 ) ])
    #A_sub_diag = np.diag(sub_diag, k=-1)

    #A = A_diag + A_sup_diag + A_sub_diag

    #construct Phi
    Phi = np.empty_like( grid )
    Phi[ 0 ] = alpha
    Phi[ -1 ] = beta

    Phi[ 1:-1 ] = np.array([ l( j )
                             for j in range( 1, grid.size - 1 ) ] )

    #we can now use the banded matrix solver.
    A_banded = np.vstack(
        (
            sup_diag,
            main_diag,
            sub_diag
        )
    )

    #return linalg.solveh_banded( (int(np.floor(grid.size/2)) , int(np.floor(grid.size/2))), A, Phi)
    #return linalg.solve(A, Phi)
    #return linalg.inv(A)@Phi

    return solve_banded( ( 1, 1 ),
                         A_banded,
                         Phi
                         )


#basis function evaluation will use for plotting solutions and
#evaluating numerical solutions
def basis_eval( xs, grid, i ):
    xs = np.atleast_1d( xs )
    partition = np.array([ ( x - grid[ i - 1 ] )  / ( grid[ i ] - grid[ i - 1 ] )
                           if ( i != 0 and grid[ i - 1 ] <= x and x <= grid[ i ] )
                           else ( grid[ i + 1 ] - x ) / ( grid[ i + 1 ] - grid[ i ] )
                           if ( grid[ i ] <= x and x <= grid[ i + 1 ] )
                           else 0
                           for x in xs ])
    return partition

#returns the numerical solution
def compute_num_sol( domain, grid, alpha, beta, eps ):
    #gets the basis coefficients
    K = FiniteElement( grid,
                       alpha,
                       beta,
                       eps )
    #returns the numerical solution array
    return np.array([ k * basis_eval( domain,
                                      grid,
                                      i
                                      )
                      for i, k in enumerate( K ) ]).sum( axis=0 )


#analytic solution function
def analytic_sol( x, a=2, b=4, eps=1/50. ):

    return a + x + ( b - a -1 ) * ( ( np.exp( x / eps ) - 1 ) / ( np.exp( 1 / eps ) -1 ) )

def test_basis():
    grid = np.linspace( 0, 1, 6 )
    domain = np.linspace( 0, 1, 300 )
    fig = plt.figure()
    fig.set_dpi( 150 )
    ax = fig.add_subplot( 111 )
    for i in range( 6 ):
        ax.plot( domain,
                 basis_eval( domain,
                             grid,
                             i
                             ),
                 label='Basis Function i = ' + str( i )
                 )
        ax.legend( loc='best' )

    ax.set_title( 'Piecewise Linear Polynomial Basis' )
    plt.show()

    return


def problem1():
    #set domain and grid
    domain = np.linspace(
        0,
        1,
        1000
    )
    grid = np.linspace(
        0,
        1,
        101
    )
    #get numerical solution
    alpha, beta, eps = 2, 4, 1/50.
    num_sol = compute_num_sol(
        domain,
        grid,
        alpha,
        beta,
        eps
    )

    #plot analytic solution and numerical solution
    fig, ax = plt.subplots( 1, 1 )
    fig.set_dpi( 150 )
    ax.plot( domain,
             analytic_sol(
                 domain
             ),
             'r-',
             linewidth=3.5,
             label='Analytic Solution'
             )
    ax.plot(
        domain,
        num_sol,
        'b-',
        label='Numerical Solution using FEA'
    )
    ax.legend( loc='best' )
    ax.set_xlabel( 'x' )
    ax.set_ylabel( 'y' )
    ax.set_title( 'Problem 1' )
    plt.show()

    return

def problem2():
    #set domain and grid
    domain = np.linspace(
        0,
        1,
        1000
    )
    even_grid = np.linspace(
        0,
        1,
        15
    )
    clustered_grid = np.linspace(
        0,
        1,
        15
    )**( 1/8. )
    #get numerical solution
    alpha, beta, eps = 2, 4, 1/50.
    even_num_sol = compute_num_sol(
        domain,
        even_grid,
        alpha,
        beta,
        eps
    )
    clustered_num_sol = compute_num_sol(
        domain,
        clustered_grid,
        alpha,
        beta,
        eps
    )

    #now plot and compare
    fig = plt.figure()
    fig.set_dpi( 150 )
    ax = fig.add_subplot( 111 )
    ax.plot(
        domain,
        even_num_sol,
        'r-',
        label='Even Grid'
    )
    ax.plot(
        domain,
        clustered_num_sol,
        'b-',
        label='Clustered Grid'
    )
    ax.legend( loc='best' )
    ax.set_xlabel( 'x' )
    ax.set_ylabel( 'y' )
    ax.set_title( 'Problem 2' )
    return

def FEA_parallel():

    dview = initialize()
    dview.execute('import numpy as np')
    dview.execute('from scipy.linalg import solve_banded')
    dview.execute('from scipy import linalg')

    def get_numerical_errors(ns):

        def FiniteElement(grid, alpha, beta, eps):
            '''
            :param grid:
            :param alpha:
            :param beta:
            :param eps:
            :return:
            '''

            # helper function to construct A matrix
            def a(i, j):
                if j == i + 1:
                    hp1 = grid[i + 1] - grid[i]
                    return eps / hp1 + 1 / 2.
                elif j == i:
                    h1 = grid[i] - grid[i - 1]
                    hp1 = grid[i + 1] - grid[i]
                    return -eps / h1 - eps / hp1
                elif j == i - 1:
                    h1 = grid[i] - grid[i - 1]
                    return eps / h1 - 1 / 2.
                else:
                    return 0

            # helper funtion to construct Phi vector
            def l(j):
                h1 = grid[j] - grid[j - 1]
                hp1 = grid[j + 1] - grid[j]
                return -1 * (h1 + hp1) / 2

            # construct A
            main_diag = np.ones_like(grid)
            main_diag[1:-1] = np.array([a(i, i) for i in range(1, grid.size - 1)])

            sup_diag = np.zeros(grid.size)
            sup_diag[2:] = np.array([a(i + 1, i) for i in range(1, grid.size - 1)])

            sub_diag = np.zeros(grid.size)
            sub_diag[:-2] = np.array([a(i, i + 1) for i in range(0, grid.size - 2)])

            # construct Phi
            Phi = np.empty_like(grid)
            Phi[0] = alpha
            Phi[-1] = beta

            Phi[1:-1] = np.array([l(j) for j in range(1, grid.size - 1)])

            # we can now use the banded matrix solver.
            A_banded = np.vstack((sup_diag, main_diag, sub_diag))


            return solve_banded((1, 1), A_banded, Phi)

        # basis function evaluation will use for plotting solutions and
        # evaluating numerical solutions
        def basis_eval(xs, grid, i):
            xs = np.atleast_1d(xs)
            partition = np.array([(x - grid[i - 1]) / (grid[i] - grid[i - 1])
                                  if (i != 0 and grid[i - 1] <= x and x <= grid[i])
                                  else (grid[i + 1] - x) / (grid[i + 1] - grid[i])
            if (grid[i] <= x and x <= grid[i + 1])
            else 0 for x in xs])
            return partition

        def analytic_sol(x, a=2, b=4, eps=1 / 50.):
            return a + x + (b - a - 1) * ((np.exp(x / eps) - 1) / (np.exp(1 / eps) - 1))

        def compute_num_sol(domain, grid, alpha, beta, eps):
            # gets the basis coefficients
            K = FiniteElement(grid, alpha, beta, eps)
            # returns the numerical solution array
            return np.array([k * basis_eval(domain, grid, i) for i, k in enumerate(K)]).sum(axis=0)

        domain = np.linspace(0, 1, 1000)
        true_sol = analytic_sol(domain)
        alpha, beta, eps = 2, 4, 1 / 50.
        Error = []
        ns = np.atleast_1d(ns)

        if ns.size == 1:
            grid = np.linspace(0, 1, 2**(ns[0]))
            curr_sol = compute_num_sol(domain, grid, alpha, beta, eps)
            curr_error = linalg.norm(curr_sol - true_sol, ord = np.inf)
            return curr_error

        else:
            for n in ns:
                grid = np.linspace(0, 1, 2**n)
                curr_sol = compute_num_sol(domain, grid, alpha, beta, eps)
                curr_error = linalg.norm(curr_sol - true_sol, ord = np.inf)
                Error.append(curr_error)

            return Error

    partition_n_vals = [[4, 5], [6, 20], [7, 19], [8, 18], [9, 17, 10], [11, 16, 12], [13, 15, 14], [21]]

    response = dview.map_async(get_numerical_errors, partition_n_vals)
    errors = response.get()

    errors_s = np.array([
        errors[0][0], #4
        errors[0][1], #5
        errors[1][0], #6
        errors[2][0], #7
        errors[3][0], #8
        errors[4][0], #9
        errors[4][-1], #10
        errors[5][0], #11
        errors[5][-1], #12
        errors[6][0], #13
        errors[6][-1], #14
        errors[6][1], #15
        errors[5][1], #16
        errors[4][1], #17
        errors[3][1], #18
        errors[2][1], #19
        errors[1][1], #20
        errors[7][0] #21
    ])

    return errors_s

def problem3_parallel():

    n_vals = np.arange(4, 22)
    Error = FEA_parallel()

    fig = plt.figure()
    fig.set_dpi(150)
    ax = fig.add_subplot(111)
    ax.loglog(n_vals, Error)
    ax.set_ylabel(r'$E(n)$')
    ax.set_xlabel(r'$n$')
    ax.set_xlabel('Numerical Round-off Error')
    plt.show()

    return

def problem3():
    domain = np.linspace(
        0,
        1,
        1000
    )

    alpha, beta, eps = 2, 4, 1/50.
    true_sol = analytic_sol( domain )

    n_vals = np.arange(
        4,
        21
    )
    error_even = []
    for n in n_vals:
        print( n )
        #calculate error from evenly spaced solution
        grid = np.linspace(
            0,
            1,
            2**n
        )
        curr_sol = compute_num_sol(
            domain,
            grid,
            alpha,
            beta,
            eps
        )
        curr_error = linalg.norm(
            curr_sol - true_sol,
            ord=np.inf
        )
        error_even.append( curr_error )

    fig = plt.figure()
    fig.set_dpi( 150 )
    ax = fig.add_subplot( 111 )
    ax.loglog( n_vals,
               error_even,
               'k-',
               label='Error Evenly Spaced'
               )
    ax.legend( loc='best' )
    ax.set_ylabel( r'$E(n)$' )
    ax.set_xlabel( r'$n$' )
    ax.set_title( 'Numerical Round-off Error' )
    plt.show()

    return

if __name__ == "__main__":

    #test_basis()
    # problem1()
    # problem2()
    problem3()

    #problem3_parallel()