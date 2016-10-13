#include <vector>














/* TODO  - empty else if statement??
/* Takes dot product of sparse vector and dense vector.                     */
/*      + size1  - number of nonzero elements in sparse vector              */
/*      + ind1   - array of indices of nonzero elements in sparse vector    */
/*      + value1 - nonzero values in sparse vector                          */
/*      + size2  - size of dense vector                                     */
/*      + value2 - list of values in dense vector                           */
void sparse_add(vector<I> &outInd[], vector<T> &outData[], 
                const I &size1, const I ind1[], const T value1[],
                const I &size2, const I ind2[], const T value2[],
                const F &scale = 1.0)
{
    T result = 0.0;
    I lowerInd = 0;

    // Loop over elements in sparse vector, 
    for (I k=0; k<size1; k++) {
        for (I j=lowerInd; j<size2; j++) {
            // If indices overlap, add product to dot product
            if ( ind1[k] == ind2[j] ) {
                result += scale * value1[k] * value2[j];
                lowerInd = j+1;
                break;
            }
            else if( ind1[k] > ind2[j] ) {

            }

            // If inner loop vector index > outer loop vector index (assuming sorted indices),
            // the outer loop vector index is not contained in inner loop indices. 
            else if ( ind1[k] < ind2[j] ) {
                break;
            }
        }
    }
}

/* Returns dot product of sparse vector and dense vector.                   */
/*      + size1  - number of nonzero elements in sparse vector              */
/*      + ind1   - array of indices of nonzero elements in sparse vector    */
/*      + value1 - nonzero values in sparse vector                          */
/*      + size2  - size of dense vector                                     */
/*      + value2 - list of values in dense vector                           */
T sparse_dot(const I &size1, const I ind1[], const T value1[],
             const I &size2, const I ind2[], const T value2[],
             const F &scale = 1.0)
{
    T result = 0.0;
    I lowerInd = 0;
    // Loop over elements in sparse vector, 
    for (I k=0; k<size1; k++) {
        for (I j=lowerInd; j<size2; j++) {
            // If indices overlap, add product to dot product
            if ( ind1[k] == ind2[j] ) {
                result += scale * value1[k] * value2[j];
                lowerInd = j+1;
                break;
            }
            // If inner loop vector index > outer loop vector index (assuming sorted indices),
            // the outer loop vector index is not contained in inner loop indices. 
            else if ( ind1[k] < ind2[j] ) {
                break;
            }
        }
    }
    return result;
}



