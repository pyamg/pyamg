
#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <iomanip>



inline int signof(int a) { return (a<0 ? -1 : 1); }
inline float signof(float a) { return (a<0.0 ? -1.0 : 1.0); }
inline double signof(double a) { return (a<0.0 ? -1.0 : 1.0); }



/* Given an array of C-points, generate a vector separating
 * C and F-points. F-points are enumerated using 1-indexing
 * and negative numbers, while C-points are enumerated using
 * 1-indexing and positive numbers, in a vector splitting. 
 *
 * Parameters 
 * ----------
 *         Cpts : int array
 *             Array of Cpts
 *        numCpts : &int
 *            Length of Cpt array
 *        n : &int 
 *            doubleotal number of points
 *
 * Returns
 * -------
 *         splitting - vector
 *            Vector of size n, indicating whether each point is a
 *            C or F-point, with corresponding index.
 *
 */
std::vector<int> get_ind_split(const int Cpts[],
                               const int & numCpts,
                               const int &n)
{
    std::vector<int> ind_split(n,0);
    for (int i=0; i<numCpts; i++) {
        ind_split[Cpts[i]] = 1;
    }
    int find = 1;
    int cind = 1;
    for (int i=0; i<n; i++) {
        if (ind_split[i] == 0) {
            ind_split[i] = -find;
            find += 1;
        }
        else {
            ind_split[i] = cind;
            cind += 1;
        }
    }
    return ind_split;
}


/* Generate column pointer for extracting a CSC submatrix from a
 * CSR matrix. Returns the maximum number of nonzeros in any column,
 * and teh col_ptr is modified in place.
 *
 * Parameters
 * ----------
 *
 *
 *
 *
 * Returns
 * -------
 *
 *
 */
int get_col_ptr(const int A_rowptr[],
                const int A_colinds[],
                const int &n,
                const int is_col_ind[],
                const int is_row_ind[],
                int colptr[], 
                const int &num_cols,
                const int &row_scale = 1,
                const int &col_scale = 1 )
{

    // Count instances of each col-ind submatrix
    for (int i=0; i<n; i++) {
        
        // Continue to next iteration if this row is not a row-ind
        if ( (row_scale*is_row_ind[i]) <= 0 ) {
            continue;
        }

        // Find all instances of col-inds in this row. Increase
        // column pointer to count total instances.
        //     - Note, is_col_ind[] is one-indexed, not zero.
        for (int k=A_rowptr[i]; k<A_rowptr[i+1]; k++) {
            int ind = col_scale * is_col_ind[A_colinds[k]];
            if ( ind > 0) {
                colptr[ind] += 1;
            }
        }
    }

    // Cumulative sum column pointer to correspond with data entries
    int max_nnz = 0;
    for (int i=1; i<=(num_cols); i++) {
        if (colptr[i] > max_nnz) {
            max_nnz = colptr[i];
        }
        colptr[i] += colptr[i-1];
    }
    return max_nnz;
}


/* Generate row_indices and data for CSC submatrix with col_ptr
 * determined in get_col_ptr(). Arrays are modified in place.
 *
 * Parameters 
 * ----------
 *
 *
 * Returns 
 * -------
 *
 *
 */
void get_csc_submatrix(const int A_rowptr[],
                       const int A_colinds[],
                       const double A_data[],
                       const int &n,
                       const int is_col_ind[],
                       const int is_row_ind[],
                       int colptr[], 
                       int rowinds[], 
                       double data[],
                       const int &num_cols,
                       const int &row_scale = 1,
                       const int &col_scale = 1 )
{
    // Fill in rowinds and data for sparse submatrix
    for (int i=0; i<n; i++) {
        
        // Continue to next iteration if this row is not a row-ind
        if ( (row_scale*is_row_ind[i]) <= 0 ) {
            continue;
        }

        // Find all instances of col-inds in this row. Save row-ind
        // and data value in sparse structure. Increase column pointer
        // to mark where next data index is. Will reset after.
        //     - Note, is_col_ind[] is one-indexed, not zero.
        for (int k=A_rowptr[i]; k<A_rowptr[i+1]; k++) {
            int ind = col_scale * is_col_ind[A_colinds[k]];
            if (ind > 0) {
                int data_ind = colptr[ind-1];
                rowinds[data_ind] = std::abs(is_row_ind[i])-1;
                data[data_ind] = A_data[k];
                colptr[ind-1] += 1;
            }
        }
    }

    // Reset colptr for submatrix
    int prev = 0;
    for (int i=0; i<num_cols; i++) {
        int temp = colptr[i];
        colptr[i] = prev;
        prev = temp;
    }    
}



/* Form interpolation operator using ben ideal interpolation. 
 * 
 * Parameters
 * ----------
 *        A_rowptr : int array
 *            Row pointer for A stored in CSR format.
 *        A_colinds : int array
 *            Column indices for A stored in CSR format.
 *        A_data : double array
 *            Data for A stored in CSR format.
 *        S_rowptr : int array
 *            Row pointer for sparsity pattern stored in CSR format.
 *        S_colinds : int array
 *            Column indices for sparsity pattern stored in CSR format.
 *        P_rowptr : int array
 *            Empty row pointer for interpolation operator stored in
 *            CSR format.
 *        B : double array
 *            doublearget bad guy vectors to be included in range of
 *            interpolation.
 *        Cpts : int array
 *            List of designated Cpts.
 *        n : int
 *            Degrees of freedom in A.
 *        num_bad_guys : int
 *            Number of target bad guys to include in range of
 *            interpolation.
 *
 * Returns
 * -------
 *         - An SdoubleD pair of vector<int> and vector<double>, where the
 *          vector<int> contains column indices for P in a CSR format,
 *          and the vector<double> corresponding data. In Python, this
 *           comes out as a length two tuple of tuples, where the inner
 *           tuples are the column indices and data, respectively. 
 *         - doublehe row pointer for P is modified in place.
 *
 * Notes
 * -----
 * It is important that A has sorted indices before calling this
 * function, and the list of Cpts is sorted. 
 *
 */
//     - doubleODO : test middle section with new Acf submatrix
//        --> Use get_sub_mat testing function in sparse.cpp
//
// std::pair<std::vector<int>, std::vector<double> > 
void ben_ideal_interpolation(const int A_rowptr[], const int A_rowptr_size,
                             const int A_colinds[], const int A_colinds_size,
                             const double A_data[], const int A_data_size,
                             const int S_rowptr[], const int S_rowptr_size,
                             const int S_colinds[], const int S_colinds_size,
                             int P_rowptr[], const int P_rowptr_size,
                             int P_colinds[], const int P_colinds_size,
                             double P_data[], const int P_data_size,
                             const double B[], const int B_size,
                             const int Cpts[], const int Cpts_size,
                             const int n,
                             const int num_bad_guys )
{
    std::cout << "started\n";

    /* ------ tested ----- */
    // Get splitting of points in one vector, Cpts enumerated in positive,
    // one-indexed ordering and Fpts in negative. 
    //         E.g., [-1,1,2,-2,-3] <-- Fpts = [0,3,4], Cpts = [1,2]
    std::vector<int> splitting = get_ind_split(Cpts,Cpts_size,n);

    // Get sparse CSC column pointer for submatrix Acc. Final two arguments
    // positive to select (positive indexed) C-points for rows and columns.
    std::vector<int> Acc_colptr(Cpts_size+1,0);
    get_col_ptr(A_rowptr, A_colinds, n, &splitting[0],
                &splitting[0], &Acc_colptr[0],
                Cpts_size, 1, 1);

    // Allocate row-ind and data arrays for sparse submatrix 
    int nnz = Acc_colptr[Cpts_size];
    std::vector<int> Acc_rowinds(nnz,0);
    std::vector<double> Acc_data(nnz,0);

    // Fill in sparse structure for Acc. 
    get_csc_submatrix(A_rowptr, A_colinds, A_data, n, &splitting[0],
                      &splitting[0], &Acc_colptr[0], &Acc_rowinds[0],
                      &Acc_data[0], Cpts_size, 1, 1);

    // Form constraint vector, \hat{B}_c = A_{cc}B_c, in column major
    std::vector<double> constraint(num_bad_guys*Cpts_size, 0);
    for (int j=0; j<Cpts_size; j++) {
        for (int k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
            for (int i=0; i<num_bad_guys; i++) {
                constraint[col_major(Acc_rowinds[k],i,Cpts_size)] += 
                                    Acc_data[k] * B[col_major(Cpts[j],i,n)];
            }
        }
    }

    // Get sparse CSR submatrix Acf. First estimate number of nonzeros
    // in Acf and preallocate arrays. 
    int Acf_nnz = 0;
    for (int i=0; i<Cpts_size; i++) {
        int temp = Cpts[i];
        Acf_nnz += A_rowptr[temp+1] - A_rowptr[temp];
    }
    Acf_nnz *= (n - Cpts_size) / n;

    std::vector<int> Acf_rowptr(Cpts_size+1,0);
    std::vector<int> Acf_colinds;
    Acf_colinds.reserve(Acf_nnz);
    std::vector<double> Acf_data;
    Acf_data.reserve(Acf_nnz);

    // Loop over the row for each C-point
    for (int i=0; i<Cpts_size; i++) {
        int temp = Cpts[i];
        int nnz = 0;
        // Check if each col_ind is an F-point (splitting < 0)
        for (int k=A_rowptr[temp]; k<A_rowptr[temp+1]; k++) {
            int col = A_colinds[k];
            // If an F-point, store data and F-column index. Note,
            // F-index is negative and 1-indexed in splitting. 
            if (splitting[col] < 0) {
                Acf_colinds.push_back(abs(splitting[col]) - 1);
                Acf_data.push_back(A_data[k]);
                nnz += 1;
            }
        }
        Acf_rowptr[i+1] = Acf_rowptr[i] + nnz;
    }

    // Get maximum number of rows selected in minimization submatrix
    // (equivalent to max columns per row in sparsity pattern for W).
    int max_rows = 0;
    for (int i=1; i<S_rowptr_size; i++) {
        int temp = S_rowptr[i]-S_rowptr[i-1];
        if (max_rows < temp) {
            max_rows = temp;
        } 
    }

    // Get maximum number of nonzero columns per row in Acf submatrix.
    int max_cols = 0;
    for (int i=0; i<Cpts_size; i++) {
        int temp = Acf_rowptr[i+1] - Acf_rowptr[i];
        if (max_cols < temp) {
            max_cols = temp;
        }
    }

    // Preallocate storage for submatrix used in minimization process
    // Generally much larger than necessary, but may be needed in certain
    // cases. 
    int max_size = max_rows * (max_rows * max_rows); 
    std::vector<double> sub_matrix(max_size, 0);

    // Allocate pair of vectors to store P.col_inds and P.data. Use
    // size of sparsity pattern as estimate for number of nonzeros. 
    // Pair returned by the function through SWIG.
    // std::pair<std::vector<int>, std::vector<double> > P_vecs;
    // std::get<0>(P_vecs).reserve(S_colinds_size);
    // std::get<1>(P_vecs).reserve(S_colinds_size);

    /* ------------------ */

    std::cout << "starting loop - submat size = " << max_size << "\n";

    // Form P row-by-row
    P_rowptr[0] = 0;
    int numCpts = 0;
    int data_ind = 0;
    for (int row_P=0; row_P<n; row_P++) {

        std::cout << row_P << std::endl;

        // Check if row is a C-point (>0 in splitting vector).
        // If so, add identity to P. Recall, enumeration of C-points
        // in splitting is one-indexed. 
        if (splitting[row_P] > 0) {
            // std::get<0>(P_vecs).push_back(splitting[row_P]-1);
            // std::get<1>(P_vecs).push_back(1.0);
            P_rowptr[row_P+1] = P_rowptr[row_P] + 1;
            P_colinds[data_ind] = splitting[row_P]-1;
            P_data[data_ind] = 1.0;
            data_ind += 1;
            numCpts +=1 ;
        }

        // If row is an F-point, form row of \hat{W} through constrained
        // minimization and multiply by A_{cc} to get row of P. 
        else {

            // Find row indices for all nonzero elements in submatrix of Acf.
            std::set<int> col_inds;

            // Get number of columns in sparsity pattern (rows in submatrix),
            // create pointer to indices
            const int *row_inds = &S_colinds[S_rowptr[row_P]];
            int submat_m = S_rowptr[row_P+1] - S_rowptr[row_P];
            
            // Get all nonzero col indices of any row in sparsity pattern
            for (int j=0; j<submat_m; j++) {
                int temp_row = row_inds[j];
                for (int i=Acf_rowptr[temp_row]; i<Acf_rowptr[temp_row+1]; i++) {
                    col_inds.insert(Acf_colinds[i]);
                }
            }
            int submat_n = col_inds.size();

            // Fill in row major data array for submatrix
            int submat_ind = 0;

            if ( (submat_n == 0) || (submat_m == 0) ) {
                P_rowptr[row_P+1] = P_rowptr[row_P];
                continue;
            }

            // Loop over each row in submatrix
            for (int i=0; i<submat_m; i++) {
                int temp_row = row_inds[i];
                int temp_ind = Acf_rowptr[temp_row];

                // Loop over all column indices
                for (auto it=col_inds.begin(); it!=col_inds.end(); ++it) {
                    
                    // Initialize matrix entry to zero
                    sub_matrix[submat_ind] = 0.0;

                    // Check if each row, col pair is in Acf submatrix. Note, both
                    // sets of indices are ordered and Acf cols a subset of col_inds!
                    for (int k=temp_ind; k<Acf_rowptr[temp_row+1]; k++) {
                        if ( (*it) < Acf_colinds[k] ) {
                            break;
                        }
                        else if ( (*it) == Acf_colinds[k] ) {
                            sub_matrix[submat_ind] = Acf_data[k];
                            temp_ind = k+1;
                            break;
                        }
                    }
                    submat_ind += 1;
                }
            }

            std::cout << "\tformed submatrix, " <<  submat_m << " x " << submat_n << std::endl;
            print_mat(&sub_matrix[0],submat_m,submat_n,0);
            std::cout << "\trow inds\n\t";
            for (int i=0; i<submat_m; i++) {
                std::cout << row_inds[i] << ", ";
            }
            std::cout << "\n\tcol inds\n\t";
            for (auto it=col_inds.begin(); it!=col_inds.end(); ++it) {
                std::cout << *it << ", ";
            }
            std::cout << "\n";

            // Make right hand side basis vector for this row of W, which is
            // the current F-point.
            int f_row = row_P - numCpts;
            std::vector<double> sub_rhs(submat_n,0);
            {
                int l=0;
                for (auto it=col_inds.begin(); it!=col_inds.end(); it++, l++) {
                    if ( (*it) == f_row ) {
                        sub_rhs[l] = 1.0;
                    }
                }
            }

            std::cout << "\tformed sub rhs\n";
            print_mat(&sub_rhs[0],1,submat_n,0);

            // Restrict constraint vector to sparsity pattern
            std::vector<double> sub_constraint;
            sub_constraint.reserve(submat_m * num_bad_guys);
            for (int k=0; k<num_bad_guys; k++) {
                for (int i=0; i<submat_m; i++) {
                    int temp_row = row_inds[i];
                    sub_constraint.push_back( constraint[col_major(temp_row,k,numCpts)] );
                }
            }


            std::cout << "\tformed sub constraint\n";
            print_mat(&sub_constraint[0],1,submat_m,0);

            // Get rhs of constraint - this is just the (f_row)th row of B_f,
            // which is the (row_P)th row of B.
            std::vector<double> constraint_rhs(num_bad_guys,0);
            for (int k=0; k<num_bad_guys; k++) {
                constraint_rhs[k] = B[col_major(row_P,k,n)];
            }


            std::cout << "\tformed constraint rhs = " << constraint_rhs[0] << "\n";

            // Solve constrained least sqaures, store solution in w_l. 
            std::vector<double> w_l = constrained_least_squares(sub_matrix,
                                                                sub_rhs,
                                                                sub_constraint,
                                                                constraint_rhs,
                                                                submat_n,
                                                                submat_m,
                                                                num_bad_guys);

            std::cout << "\tsolved CLS" << std::endl;
            print_mat(&w_l[0],1,w_l.size(),0);

            /* ---------- tested ---------- */
            // Loop over each jth column of Acc, taking inner product
            //         (w_l)_j = \hat{w}_l * (Acc)_j
            // to form w_l := \hat{w}_l*Acc.
            int row_length = 0;
            for (int j=0; j<Cpts_size; j++) {
                double temp_prod = 0;
                int temp_v0 = 0;
                // Loop over nonzero indices for this column of Acc and vector w_l.
                // Note, both have ordered, unique indices.
                for (int k=Acc_colptr[j]; k<Acc_colptr[j+1]; k++) {
                    for (int i=temp_v0; i<submat_m; i++) {

                        std::cout << "Acc col " << j << ", Acc row " << Acc_rowinds[k] << ", w_l ind " << row_inds[i] << "\n";

                        // Can break here because indices are sorted increasing
                        if ( row_inds[i] > Acc_rowinds[k] ) {
                            break;
                        }
                        // If nonzero, add to dot product 
                        else if (row_inds[i] == Acc_rowinds[k]) {
                            temp_prod += w_l[i] * Acc_data[k];
                            temp_v0 += 1;
                            break;
                        }
                        else {
                            temp_v0 += 1;
                        }
                    }
                }
                // If dot product of column of Acc and vector \hat{w}_l is nonzero,
                // add to sparse structure of P.
                if (std::abs(temp_prod) > 1e-12) {
                    // std::get<0>(P_vecs).push_back(j);
                    // std::get<1>(P_vecs).push_back(temp_prod);
                    P_colinds[data_ind] = j;
                    P_data[data_ind] = temp_prod;
                    row_length += 1;
                    data_ind += 1;
                }
                std::cout << "\t\tval = " << temp_prod << "\n";
            }

            // Set row pointer for next row in P
            P_rowptr[row_P+1] = P_rowptr[row_P] + row_length;
            row_inds = NULL;

            std::cout << "\tAdded row to P" << std::endl;

        }

        if (data_ind > P_data_size) {
            std::cout << "Warning - more nonzeros in P than allocated - breaking early.\n";
            break;
        }
    }
    

    std::cout << "Finished Loop" << std::endl;



    // Check that all C-points were added to P. 
    if (numCpts != Cpts_size) {
        std::cout << "Warning - C-points missed in constructing P.\n";
    }

}


int main(int argc, char *argv[])
{

    std::vector<int> A_rowptr {0,2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,58};
    std::vector<int> A_colinds {0,1,0,1,2,1,2,3,2,3,4,3,4,5,4,5,6,5,6,7,6,7,8,7,8,9,8,9,
                                10,9,10,11,10,11,12,11,12,13,12,13,14,13,14,15,14,15,16,
                                15,16,17,16,17,18,17,18,19,18,19};
    std::vector<double> A_data {2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,
                                2,-1,-1, 2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,
                                2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1,2};
    std::vector<int> S_rowptr {0,1,3,5,5,7,9,9,11,13,13,15,17,17,19,21,21,23,25,25,26};
    std::vector<int> S_colinds {0,0,1,0,1,1,2,1,2,2,3,2,3,3,4,3,4,4,5,4,5,5,6,5,6,6};
    std::vector<int> Cpts {0,3,6,9,12,15,18};
 
    int numCpts = Cpts.size();
    int n = A_rowptr.size()-1;
    int num_bad_guys = 1;
    int max_size = S_colinds.size()+numCpts;

    std::vector<double> B (n,1);
    std::vector<int> P_rowptr (n+1,0);
    std::vector<int> P_colinds (max_size,0);
    std::vector<double> P_data (max_size,0);


    ben_ideal_interpolation(&A_rowptr[0], A_rowptr.size(),
                            &A_colinds[0], A_colinds.size(),
                            &A_data[0], A_data.size(),
                            &S_rowptr[0], S_rowptr.size(),
                            &S_colinds[0], S_colinds.size(),
                            &P_rowptr[0], P_rowptr.size(),
                            &P_colinds[0], P_colinds.size(),
                            &P_data[0], P_data.size(),
                            &B[0], B.size(),
                            &Cpts[0], Cpts.size(),
                             n,
                             num_bad_guys );

    std::cout << "P_rowptr = \n\t";
    for (auto it=P_rowptr.begin(); it!=P_rowptr.end(); ++it) {
        std::cout << *it << ", ";
    }
    std::cout << "\nP_colinds = \n\t";
    for (auto it=P_colinds.begin(); it!=P_colinds.end(); ++it) {
        std::cout << *it << ", ";
    }
    std::cout << "\nP_data = \n\t";
    for (auto it=P_data.begin(); it!=P_data.end(); ++it) {
        std::cout << *it << ", ";
    }




}





