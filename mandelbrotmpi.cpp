#include <iostream>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <stdio.h>
#include <mpi.h>

using namespace std;
#define MASTeR 0

double mapToReal(int, int, double, double);
double mapToImaginary(int, int, double, double);
int findMandelbrot(double, double, int);

/**
 * Implementing the pseudocode:
 * For each pixel (Px, Py) on the screen, do:
    {
  x0 = scaled x coordinate of pixel (scaled to lie in the Mandelbrot X scale (-2.5, 1))
  y0 = scaled y coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1, 1))
  x = 0.0
  y = 0.0
  iteration = 0
  max_iteration = 1000
  while (x*x + y*y <= 2*2  AND  iteration < max_iteration) {
    xtemp = x*x - y*y + x0
    y = 2*x*y + y0
    x = xtemp
    iteration = iteration + 1
  }
  color = palette[iteration]
  plot(Px, Py, color)
    }
 *
 */

int main(int argc, char** argv)
{

    int rank, numProcess;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);



    /**
     * Se tiene un documento txt en la carpeta root con las variables
     * Esto permite editar mas facil los valores de input del fractal
     */


    int width, height, maxN;
    double minR, maxR, minI, maxI;
    double stime, etime;
    double cr, ci;
    int n;

    stime = MPI_Wtime();

    if (rank == MASTeR)
    {
        ifstream fin("input");
        if (!fin)
        {
            cout << "Could not open the file" << endl;
            cin.ignore();
            return 0;
        }

        fin >> width >> height >> maxN;
        fin >> minR >> maxR >> minI >> maxI;
        fin.close();
    }

    MPI_Bcast(&width, 1, MPI_INT, MASTeR, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, MASTeR, MPI_COMM_WORLD);
    MPI_Bcast(&maxN, 1, MPI_INT, MASTeR, MPI_COMM_WORLD);
    MPI_Bcast(&minR, 1, MPI_DOUBLE, MASTeR, MPI_COMM_WORLD);
    MPI_Bcast(&maxR, 1, MPI_DOUBLE, MASTeR, MPI_COMM_WORLD);
    MPI_Bcast(&minI, 1, MPI_DOUBLE, MASTeR, MPI_COMM_WORLD);
    MPI_Bcast(&maxI, 1, MPI_DOUBLE, MASTeR, MPI_COMM_WORLD);

    int numTasksPerProc = height / numProcess;
    int mod = height % numProcess;


    int start_height = rank * numTasksPerProc;
    int end_height = start_height + numTasksPerProc;


    // master will process additional rows
    // for example if height is 9 and processes are 4
    // then rank 0 (master) == 3 tasks   start = 0 , end = 3 (exclusive)
    // and rank 1  = 2 tasks   start = 3 , end = 5 (exclusive)
    // and rank 2  = 2 tasks   start = 5 , end = 7 (exclusive)
    // and rank 3  = 2 tasks   start = 7 , end = 9 (exclusive)
    if (rank == MASTeR)
    {
        end_height += mod;
    }
    else
    {
        start_height += mod;
        end_height += mod;
    }


    /**
     * Se abre el archivo de output y se escribe el header PPM
     */
    ofstream fout;

    if (rank == MASTeR)
    {
        fout.open("mandelbrot_mpi.ppm");
        fout << "P3" << endl;
        fout << width << " " << height << endl; //dimensiones
        fout << "256" << endl;                  //maximo valor del RGB
        //Para cada pixel
        //private(cr, ci, n)
    }


    int* r_values = new int[height * width];
    int* g_values = new int[height * width];
    int* b_values = new int[height * width];

    for (int y = start_height; y < end_height; y++) //Filas
    {
        for (int x = 0; x < width; x++) //Pixeles en la fila
        {
            //Se encuentran las partes reales e imaginarias de C

            cr = mapToReal(x, width, minR, maxR);
            ci = mapToImaginary(y, height, minI, maxI);
            //Se encuentran el numero de iteraciones en la formula de Mandelbrot
            n = findMandelbrot(cr, ci, maxN);
            //Se mapean los numeros resultantes a un valor RGB
            int r = n % 256; //este numero se puede variar para la coloracion
            int g = n % 256; //este numero se puede variar para la coloracion
            int b = n % 256; //este numero se puede variar para la coloracion
            //Se pasan estos valores a la imagen
            r_values[y * width + x] = r;
            g_values[y * width + x] = g;
            b_values[y * width + x] = b;
        }
    }


    int* r_values_gather, *g_values_gather, *b_values_gather;

    if (rank == MASTeR)
    {
        r_values_gather = new int[height * width];
        g_values_gather = new int[height * width];
        b_values_gather = new int[height * width];
    }


    MPI_Gather(&r_values[start_height * width], numTasksPerProc * width, MPI_INT, &r_values_gather[start_height * width], numTasksPerProc * width, MPI_INT, MASTeR, MPI_COMM_WORLD);

    MPI_Gather(&g_values[start_height * width], numTasksPerProc * width, MPI_INT, &b_values_gather[start_height * width], numTasksPerProc * width, MPI_INT, MASTeR, MPI_COMM_WORLD);

    MPI_Gather(&b_values[start_height * width], numTasksPerProc * width, MPI_INT, &g_values_gather[start_height * width], numTasksPerProc * width, MPI_INT, MASTeR, MPI_COMM_WORLD);


    if (rank == MASTeR)
    {


        // all other rows that are gathered
        for (int i = 0; i < height * width; i++)
        {
            fout << r_values_gather[i] << " " << g_values_gather[i] << " " << b_values_gather[i] << " ";

            if (i % width == 0)
            {
                fout << endl;
            }
        }
        fout.close();

        delete[] r_values_gather;
        delete[] g_values_gather;
        delete[] b_values_gather;
    }

    delete[] r_values;
    delete[] g_values;
    delete[] b_values;
    etime = MPI_Wtime();

    if (rank == MASTeR)
        cout << "Time in seconds =" << etime - stime << endl;

    MPI_Finalize();
    return 0;
}

double mapToReal(int x, int width, double minR, double maxR)
{
    double range = maxR - minR;
    /**
     * [0, width]
     * [0, maxR - minR] - val * range / width
     * [minR, maxR] - last step + minR
     */
    return x * (range / width) + minR;
}

double mapToImaginary(int y, int width, double minI, double maxI)
{
    double range = maxI - minI;
    /**
     * [0, width]
     * [0, maxI - minI] - val * range / width
     * [minI, maxI] - last step + minI
     */
    return y * (range / width) + minI;
}

int findMandelbrot(double cr, double ci, int max_iterations)
{

    int i = 0;
    double zr = 0.0, zi = 0.0;
    while (i < max_iterations && (zr * zr) + (zi * zi) < 4.0)
    {
        double temp = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = temp;
        i++;
    }
    return i;
}

