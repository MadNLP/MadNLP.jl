/* PlasmoNLP.jl */
/* Created by Sungho Shin (sungho.shin@wisc.edu) */

extern void pardisoinit(void* pt, const int* mtype, const int* solver, 
			int* iparm, double* dparm, int* error);
extern void pardiso(void** pt, const int* maxfct, const int* mnum, 
		    const int* mtype, const int* phase, const int* n, 
		    const double* a, const int* ia, const int* ja, 
		    const int* perm, const int* nrhs, int* iparm, 
		    const int* msglvl, double* b, double* x, 
		    int* error, double* dparm);

void _pardisoinit_dummy(void* pt, const int* mtype, const int* solver,
			int* iparm, double* dparm, int* error) {
  pardisoinit(pt,mtype,solver,iparm,dparm,error);
}

void _pardiso_dummy(void** pt,const int* maxfct,const int* mnum,
		    const int* mtype,const int* phase,const int* n,
		    const double* a,const int* ia,const int* ja,
		    const int* perm,const int* nrhs,int* iparm,
		    const int* msglvl,double* b,double* x,
		    int* error,double* dparm) {
  pardiso(pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x,error,dparm);
}
