/*
 * Copyright (c) 2011, 2012 Tobias Kalbitz <tobias.kalbitz@googlemail.com>
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v2.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/old-licenses/gpl-2.0.html
 */

#include "pso_copy.h"

void copy_gb_rating_dev_to_host(struct pso_instance* inst, void* parent_rat_cpy)
{
	CUDA_CALL(cudaMemcpy(parent_rat_cpy,
			     inst->gb_rat,
			     BLOCKS * sizeof(double),
			     cudaMemcpyDeviceToHost));
}


void copy_globals_dev_to_host(struct pso_instance* inst, void* global_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_particle_gbest;
	p.dstPtr = make_cudaPitchedPtr(
			global_cpy,
			inst->dev_particle_gbest_ext.width,
			inst->dev_particle_gbest_ext.width / sizeof(double),
			inst->dim.matrix_height);

	p.extent = inst->dev_particle_gbest_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_particles_dev_to_host(struct pso_instance* inst, void* particle_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_particle;
	p.dstPtr = make_cudaPitchedPtr(
			particle_cpy,
			inst->dev_particle_ext.width,
			inst->dev_particle_ext.width / sizeof(double),
			inst->dim.matrix_height);

	p.extent = inst->dev_particle_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_lbest_particles_dev_to_host(struct pso_instance* inst, void* particle_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_particle_lbest;
	p.dstPtr = make_cudaPitchedPtr(
			particle_cpy,
			inst->dev_particle_lbest_ext.width,
			inst->dev_particle_lbest_ext.width / sizeof(double),
			inst->dim.matrix_height);

	p.extent = inst->dev_particle_lbest_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}

void copy_particle_rating_dev_to_host(struct pso_instance* inst,
				      void* particle_rat_cpy)
{
	cudaMemcpy3DParms p = { 0 };
	p.srcPtr = inst->dev_prat;
	p.dstPtr = make_cudaPitchedPtr(
			particle_rat_cpy,
			inst->dev_prat_ext.width,
			inst->dev_prat_ext.width / sizeof(double),
			1);

	p.extent = inst->dev_prat_ext;
	p.kind = cudaMemcpyDeviceToHost;
	CUDA_CALL(cudaMemcpy3D(&p));
}
