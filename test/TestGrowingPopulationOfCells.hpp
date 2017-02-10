/*

Copyright (c) 2005-2017, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#ifndef TESTGROWINGPOPULATIONOFCELLS_HPP_
#define TESTGROWINGPOPULATIONOFCELLS_HPP_

// Needed for the test environment
#include <cxxtest/cxxtest/TestSuite.h>
#include "AbstractCellBasedTestSuite.hpp"

#include "OffLatticeSimulation.hpp"
#include "TransitCellProliferativeType.hpp"
#include "ExponentialG1GenerationalCellCycleModel.hpp"
#include "CellsGenerator.hpp"
#include "ImmersedBoundaryMesh.hpp"
#include "ImmersedBoundarySimulationModifier.hpp"
#include "ImmersedBoundaryHoneycombMeshGenerator.hpp"
#include "ImmersedBoundaryLinearMembraneForce.hpp"
#include "ImmersedBoundaryLinearInteractionForce.hpp"
#include "FluidSource.hpp"
#include "SmartPointers.hpp"
#include "SuperellipseGenerator.hpp"
#include "SimpleTargetAreaModifier.hpp"

#include "ForwardEulerNumericalMethod.hpp"
#include <boost/make_shared.hpp>

#include "Debug.hpp"


// Simulation does not run in parallel
#include "FakePetscSetup.hpp"

class TestGrowingPopulationOfCells : public AbstractCellBasedTestSuite
{
public:

    void xTestProliferateFromSingleCell() throw(Exception)
    {
        /*
         * 1: num nodes
         * 2: superellipse exponent
         * 3: cell width
         * 4: cell height
         * 5: bottom left x
         * 6: bottom left y
         */
        SuperellipseGenerator* p_gen = new SuperellipseGenerator(64, 1.0, 0.1, 0.1, 0.45, 0.45);
        std::vector<c_vector<double, 2> > locations = p_gen->GetPointsAsVectors();

        std::vector<Node<2>* > nodes;
        std::vector<ImmersedBoundaryElement<2,2>* > elements;

        for (unsigned location = 0; location < locations.size(); location++)
        {
            nodes.push_back(new Node<2>(location, locations[location], true));
        }

        elements.push_back(new ImmersedBoundaryElement<2,2>(0, nodes));

        ImmersedBoundaryMesh<2,2>* p_mesh = new ImmersedBoundaryMesh<2,2>(nodes, elements);
        p_mesh->SetNumGridPtsXAndY(64);

        std::vector<CellPtr> cells;
        MAKE_PTR(TransitCellProliferativeType, p_cell_type);
        CellsGenerator<ExponentialG1GenerationalCellCycleModel, 2> cells_generator;
        cells_generator.GenerateBasicRandom(cells, p_mesh->GetNumElements(), p_cell_type);

        ImmersedBoundaryCellPopulation<2> cell_population(*p_mesh, cells);
        cell_population.SetIfPopulationHasActiveSources(true);

        // Loop over all cells
        for (AbstractCellPopulation<2>::Iterator cell_it = cell_population.Begin();
             cell_it != cell_population.End();
             ++cell_it)
        {
            dynamic_cast<ExponentialG1GenerationalCellCycleModel*>(cell_it->GetCellCycleModel())->SetRate(10.0);
            dynamic_cast<ExponentialG1GenerationalCellCycleModel*>(cell_it->GetCellCycleModel())->SetMaxTransitGenerations(5);
        }

        OffLatticeSimulation<2> simulator(cell_population);
        simulator.SetNumericalMethod(boost::make_shared<ForwardEulerNumericalMethod<2,2> >());
        simulator.GetNumericalMethod()->SetUseUpdateNodeLocation(true);

        // Add main immersed boundary simulation modifier
        MAKE_PTR(ImmersedBoundarySimulationModifier<2>, p_main_modifier);
        simulator.AddSimulationModifier(p_main_modifier);

        MAKE_PTR(SimpleTargetAreaModifier<2>, p_area_modifier);
        simulator.AddSimulationModifier(p_area_modifier);
        p_area_modifier->SetReferenceTargetArea(p_mesh->GetVolumeOfElement(0));

        // Add force law
        MAKE_PTR(ImmersedBoundaryLinearMembraneForce<2>, p_boundary_force);
        p_main_modifier->AddImmersedBoundaryForce(p_boundary_force);
        p_boundary_force->SetElementSpringConst(1.0 * 1e7);

        // Set simulation properties
        double dt = 0.01;
        simulator.SetOutputDirectory("TestProliferateFromSingleCell");
        simulator.SetDt(dt);
        simulator.SetSamplingTimestepMultiple(100u);
        simulator.SetEndTime(200.0);

        simulator.Solve();
    }

    void TestProliferateFromHoneycomb() throw(Exception)
    {
        ImmersedBoundaryHoneycombMeshGenerator gen(4, 3, 8, 0.025, 0.4);
        ImmersedBoundaryMesh<2, 2>* p_mesh = gen.GetMesh();
        p_mesh->SetNumGridPtsXAndY(64);

        std::vector<CellPtr> cells;
        MAKE_PTR(TransitCellProliferativeType, p_cell_type);
        CellsGenerator<ExponentialG1GenerationalCellCycleModel, 2> cells_generator;
        cells_generator.GenerateBasicRandom(cells, p_mesh->GetNumElements(), p_cell_type);

        ImmersedBoundaryCellPopulation<2> cell_population(*p_mesh, cells);
        cell_population.SetIfPopulationHasActiveSources(true);

        // Loop over all cells
        for (AbstractCellPopulation<2>::Iterator cell_it = cell_population.Begin();
             cell_it != cell_population.End();
             ++cell_it)
        {
            dynamic_cast<ExponentialG1GenerationalCellCycleModel*>(cell_it->GetCellCycleModel())->SetTransitCellG1Duration(5.0);
            dynamic_cast<ExponentialG1GenerationalCellCycleModel*>(cell_it->GetCellCycleModel())->SetMaxTransitGenerations(3);
        }

        OffLatticeSimulation<2> simulator(cell_population);
        simulator.SetNumericalMethod(boost::make_shared<ForwardEulerNumericalMethod<2,2> >());
        simulator.GetNumericalMethod()->SetUseUpdateNodeLocation(true);

        // Add main immersed boundary simulation modifier
        MAKE_PTR(ImmersedBoundarySimulationModifier<2>, p_main_modifier);
        simulator.AddSimulationModifier(p_main_modifier);

        MAKE_PTR(SimpleTargetAreaModifier<2>, p_area_modifier);
        simulator.AddSimulationModifier(p_area_modifier);
        p_area_modifier->SetReferenceTargetArea(p_mesh->GetVolumeOfElement(0));

        // Add force law
        MAKE_PTR(ImmersedBoundaryLinearMembraneForce<2>, p_boundary_force);
        p_main_modifier->AddImmersedBoundaryForce(p_boundary_force);
        p_boundary_force->SetElementSpringConst(1.0 * 1e7);

        // Set simulation properties
        double dt = 0.01;
        simulator.SetOutputDirectory("TestProliferateFromHoneycomb");
        simulator.SetDt(dt);
        simulator.SetSamplingTimestepMultiple(100u);
        simulator.SetEndTime(10.0);

        simulator.Solve();

        PRINT_VECTOR(p_mesh->GetNode(23u)->rGetNeighbours());
    }
};


#endif /*TESTGROWINGPOPULATIONOFCELLS_HPP_*/
