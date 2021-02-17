#!/usr/bin/python3
### Importing Modules ########################################################

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
import sys
from datetime import datetime,timedelta
import time
import brainspace
from multiprocessing import Process,Queue


## my own imports
sys.path.append("../imports")
import load_data_and_functions as ldf


class gradient:

    def __init__(self):
        self.session=[]
        # Default concurrency
        self.concurrency = self.get_cpus()

    def set_kernels(self,kernels):
        self.kernels = kernels

    def set_alignment(self,alignment):
        self.alignment = alignment
    
    def set_global_alignment(self, global_alignment, extractions):
        self.global_alignment = global_alignment
        self.extractions = extractions
    
    def set_atlas_size(self,atlas_size):
        self.atlas_size = atlas_size

    def set_sparsity(self,sparsity):
        self.sparsity = sparsity

    def set_dimension_reductions(self,dimension_reductions):
        self.dimension_reductions = dimension_reductions

    def set_concatenate(self,concatenate):
        self.concatenate = concatenate

    def set_id_method(self,id_method):
        self.id_method = id_method

    def set_num_grads(self,num_grads):
        self.num_grads = num_grads

    def set_verbose(self,verbose):
        self.verbose = verbose

    def set_debug(self,debug):
        self.debug = debug

    def get_cpus(self):
        cpus = 0
        for l in open("/proc/cpuinfo"):
            if not l.strip(): cpus += 1
        return cpus

    def set_concurrency(self,concurrency):
        self.concurrency = concurrency

    def total_iterations(self):
        return len(self.sparsity) * len(self.num_grads) * len(self.kernels) * len(self.dimension_reductions)

    def add_session(self, session_path):
        session_data, session_sub, session_connectivity_data, session_cd_transposed = ldf.load_data(session_path)
        self.session.append( {
            'data'              : session_data,
            'sub'               : session_sub,
            'connectivity_data' : session_connectivity_data,
            'cd_transposed'     : session_cd_transposed
        } )


    def set_output_file(self, output_file_template):
        self.output_file_template = output_file_template

    def output_file(self):
        return  str(self.output_file_template.format(self.concatenate, self.id_method, self.global_alignment))

    def store_result(self):
        df_accuracy = pd.DataFrame(self.accuracy)
        df_accuracy.to_csv(
            self.output_file(),
            index=False,
        )


    def dump_settings(self):
        print( "Settings")
        print( "----------------------------------------------")
        print( "Kernels              : {}".format(self.kernels) )
        print( "Alignment            : {}".format(self.alignment) )
        print( "Atlas size           : {}".format(self.atlas_size) )
        print( "Sparsity             : {}".format(self.sparsity) )
        print( "Dimension reductions : {}".format(self.dimension_reductions) )
        print( "Concatenate          : {}".format(self.concatenate) )
        print( "ID method            : {}".format(self.id_method) )
        print( "Num gradients        : {}".format(self.num_grads) )
        print( "Session              : {}".format(len(self.session)) )
        print( "Total iterations     : {}".format( self.total_iterations() ))
        print( "Concurrency          : {}".format( self.concurrency ))
        print( "Output file          : {}".format( self.output_file()  ) )
        print( "----------------------------------------------")


    def reset_result_structure(self):
        # Reset output
        self.iteration_count = 0
        self.accuracy = {
            "kernels":              [],
            "dimension reduction":  [],
            "accuracy":             [],
            "n_gradients":          [],
            "sparsity":             [],
            "concatenation":        [],
        }

    def add_to_result(self,result):
        self.accuracy["kernels"].append(result["kernels"])
        self.accuracy["dimension reduction"].append(result["dimension reduction"])
        self.accuracy["accuracy"].append(result["accuracy"])
        self.accuracy["n_gradients"].append(result["n_gradients"])
        self.accuracy["sparsity"].append(result["sparsity"])
        self.accuracy["concatenation"].append(result["concatenation"])


    def run(self):
        print( "Starting..." )
        self.reset_result_structure()

        totalstarttime =time.time()
        for spars in self.sparsity:
            for n_gradients in self.num_grads:
                for kernel in self.kernels:
                    for dimension_reduction in self.dimension_reductions:
                        self.iteration_count+=1
                        if (self.verbose):
                            print("Iteration {} of {}".format(self.iteration_count, self.total_iterations()))
                        starttime = time.time()
                        self.add_to_result( self.identify_gradient( spars, n_gradients, kernel, dimension_reduction) )
                        stoptime =time.time()
                        totaltime = time.time() - totalstarttime

                        if ( self.verbose):
                            print("(this round took: {} sec".format(stoptime-starttime))
                            print("(     total took: {} sec".format(totaltime))
                            print("(  avg per round: {} sec".format(totaltime/self.iteration_count))



    def multi_worker(self,pid,q_in,q_out,total_wallclock_start):
        print( "Worker {}: starting".format(pid))


        while ( not q_in.empty() ):
            ( iteration,spars,n_gradients,kernel,dimension_reduction) = q_in.get()

            print( "Worker {}: executing iteration {}".format(pid,iteration))
            start = time.time()
            result = self.identify_gradient( spars, n_gradients, kernel, dimension_reduction)
            worktime = time.time()-start
            print( "Worker {}: finishing iteration {} in {:10.0f} sec".format(pid,iteration, worktime))
            q_out.put([result, worktime ])

            # Estimate rest time
            total       = self.total_iterations()
            todo        = q_in.qsize()
            done        = q_out.qsize()
            in_process  = total-todo-done

            elapsed = time.time() - total_wallclock_start

            remaining = (elapsed * (total-done))/done

            print( "Worker {}: Total:{} Done:{} in process:{} todo: {}".format(pid,total,done,in_process,todo))
            print( "Worker {}: projected due date {} ({:10.0f} sec left to do)".format(pid,(datetime.now() + timedelta(seconds=remaining)),remaining))



        print( "Worker {}: no iterations left, stopping".format(pid))


    def run_multi(self):
        print( "Starting multiprocess..." )
        self.reset_result_structure()

        iteration=0
        q_in  = Queue()
        for spars in self.sparsity:
            for n_gradients in self.num_grads:
                for kernel in self.kernels:
                    for dimension_reduction in self.dimension_reductions:
                        iteration+=1
                        q_in.put( [iteration, spars, n_gradients, kernel, dimension_reduction ])

        q_out = Queue()

        total_wallclock_start = time.time()

        processes =[]
        for pid in range(self.concurrency):
            p = Process(target=gradient.multi_worker, args=(self,pid,q_in, q_out,total_wallclock_start))
            processes.append(p)
            p.start()

        # Wait until output queue is filled
        while( q_out.qsize() < self.total_iterations() ):
            time.sleep(1)

        # Clean processes
        for p in processes:
            print("Joining".format(p) )
            p.join(1)


        # All wallclock
        total_wallclock = time.time() - total_wallclock_start
        total_worktime = 0
        # Collect results
        while( not q_out.empty() ):
            (result, worktime ) = q_out.get()
            self.add_to_result(result)
            total_worktime += worktime

        print( "----------------------------------------------")
        print(" Finished at:            {}".format( datetime.now() ))
        print( "Worker concurrency:     {:10.0f}".format( self.concurrency ) )
        print( "NumPy Concurrency:      {:10}".format( os.environ['OPENBLAS_NUM_THREADS'] ))
        print( "Total iterations        {:10.0f}".format( self.total_iterations() ))
        print( "Total wallclock:        {:10.0f} s".format( total_wallclock ) )
        print( "Total worktime:         {:10.0f} s".format( total_worktime ) )
        print( "Avg worktime/iteration: {:10.3f} s".format( total_worktime/self.total_iterations() ) )
        print( "Avg wallclock/iteration:{:10.3f} s".format( total_wallclock/self.total_iterations() ) )
        print( "Multiprocess speedup:   {:10.2f}".format( total_worktime/total_wallclock ) )
        print( "----------------------------------------------")




    def identify_gradient(self, spars, n_gradients, kernel, dimension_reduction ):

        if (self.debug):
            print( "    (sparsity:{}, n_gradients:{}, kernel:{} dimension_reduction:{})".format(spars, n_gradients, kernel, dimension_reduction) )

        starttime =time.time()

        ##############################################################
        ### Gradient Construction


        ## Reference Gradient for Alignment ##########################
        ## In this alignment method I will align all gradients to one
        ## reference gradient from a reference participant.
        reference_participant = ldf.get_conn_matrix(self.session[0]['cd_transposed'].iloc[:, 0])
        if self.concatenate and self.global_alignment:
            gref = GradientMaps(
                n_components    = self.extractions,
                kernel          = kernel,
                approach        = dimension_reduction,
                random_state    = 0,
            )
        else:
            gref = GradientMaps(
                    n_components    = n_gradients,
                    kernel          = kernel,
                    approach        = dimension_reduction,
                    random_state    = 0,
                )
            
        if (self.debug):
            print(" -Gradient_maps: {:10.3f} s".format( time.time() - starttime ) )
        gref.fit(reference_participant)

        if (self.debug):
            print(" -Fit:           {:10.3f} s".format( time.time() - starttime ) )

        session_gradients = []
        for session in self.session:
            session_gradients.append(
                ldf.create_gradient_database(
                    dataframe           =session['cd_transposed'],
                    subjects            =session['sub'],
                    atlas_size          =self.atlas_size,
                    reference           =gref.gradients_,
                    kernel              =kernel,
                    dimension_reduction =dimension_reduction,
                    alignment           =self.alignment,
                    n_gradients         =n_gradients,
                    concatenate         =self.concatenate,
                    global_alignment    =self.global_alignment,
                    extractions         =self.extractions
                )
            )

        if (self.debug):
            print(" -Session:           {:10.3f} s".format( time.time() - starttime ) )


        # Test all targets against all databases
        rates = []
        for target in session_gradients:
            for database in session_gradients:
                if ( target is not database ):
                    rates.append(ldf.identify(target=target, database=database))
        rate = sum(rates)/len(rates)

        if (self.debug):
            print(" -Rates:         {:10.3f} s".format( time.time() - starttime ) )



        if (self.debug):
            print(
                "Settings were " + str(kernel) + " " + str(dimension_reduction) + ", sparsity == " + str(spars) + "."
            )
            print("number of Gradients: " + str(n_gradients))
            print("       Accuracy was: " + str(rate))


        return( {
            "kernels"               :kernel,
            "dimension reduction"   :dimension_reduction,
            "accuracy"              :rate,
            "n_gradients"           :str(n_gradients),
            "sparsity"              :spars,
            "concatenation"         :self.concatenate
        })

