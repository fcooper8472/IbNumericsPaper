/*

Copyright (c) 2005-2016, University of Oxford.
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

#include <cxxtest/TestSuite.h>

// Includes from trunk
#include "CellId.hpp"
#include "CellsGenerator.hpp"
#include "CheckpointArchiveTypes.hpp"
#include "DifferentiatedCellProliferativeType.hpp"
#include "ExecutableSupport.hpp"
#include "ForwardEulerNumericalMethod.hpp"
#include "ImmersedBoundaryLinearInteractionForce.hpp"
#include "ImmersedBoundaryCellPopulation.hpp"
#include "ImmersedBoundaryLinearMembraneForce.hpp"
#include "ImmersedBoundaryMesh.hpp"
#include "ImmersedBoundaryPalisadeMeshGenerator.hpp"
#include "ImmersedBoundarySimulationModifier.hpp"
#include "OffLatticeSimulation.hpp"
#include "SmartPointers.hpp"
#include "UniformCellCycleModel.hpp"

// Boost includes
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

/*
 * Prototype functions
 */
void SetupSingletons();
void DestroySingletons();
void SetupAndRunSimulation(unsigned simulationId, unsigned numNodes);
void OutputOnCompletion(unsigned simulationId, unsigned numNodes);

int main(int argc, char *argv[])
{
    // This sets up PETSc and prints out copyright information, etc.
    ExecutableSupport::StartupWithoutShowingCopyright(&argc, &argv);

    // Define command line options
    boost::program_options::options_description general_options("This is a Chaste Immersed Boundary executable.\n");
    general_options.add_options()
                    ("help", "produce help message")
                    ("ID", boost::program_options::value<unsigned>()->default_value(0),"Index of the simulation")
                    ("NN", boost::program_options::value<unsigned>()->default_value(32),"Number of Immersed Boundary nodes");

    // Parse command line into variables_map
    boost::program_options::variables_map variables_map;
    boost::program_options::store(parse_command_line(argc, argv, general_options), variables_map);

    // Print help message if wanted
    if (variables_map.count("help"))
    {
        std::cout << setprecision(3) << general_options << "\n";
        std::cout << general_options << "\n";
        return 1;
    }

    // Get ID and name from command line
    unsigned simulation_id = variables_map["ID"].as<unsigned>();
    unsigned num_nodes = variables_map["NN"].as<unsigned>();

    SetupSingletons();
    SetupAndRunSimulation(simulation_id, num_nodes);
    DestroySingletons();
    OutputOnCompletion(simulation_id, num_nodes);
}

void SetupSingletons()
{
    // Set up what the test suite would do
    SimulationTime::Instance()->SetStartTime(0.0);

    // Reseed with 0 for same random numbers each time, or time(NULL) or simulation_id to change each realisation
    RandomNumberGenerator::Instance()->Reseed(0);
    CellPropertyRegistry::Instance()->Clear();
    CellId::ResetMaxCellId();
}

void DestroySingletons()
{
    // This is from the tearDown method of the test suite
    SimulationTime::Destroy();
    RandomNumberGenerator::Destroy();
    CellPropertyRegistry::Instance()->Clear();
}

void OutputOnCompletion(unsigned simulationId, unsigned numNodes)
{
    // Compose the message
    std::stringstream message;
    message << "Completed simulation with ID " << simulationId << " and " << numNodes << " nodes" << std::endl;

    // Send it to the console
    std::cout << message.str() << std::flush;
}
void SetupAndRunSimulation(unsigned simulationId, unsigned numNodes)
{
    /*
     * 1: num nodes
     * 2: superellipse exponent
     * 3: cell width
     * 4: cell height
     * 5: bottom left x
     * 6: bottom left y
     */
    SuperellipseGenerator* p_gen = new SuperellipseGenerator(numNodes, 1.0, 0.3, 0.6, 0.35, 0.2);
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
    MAKE_PTR(DifferentiatedCellProliferativeType, p_diff_type);
    CellsGenerator<UniformCellCycleModel, 2> cells_generator;
    cells_generator.GenerateBasicRandom(cells, p_mesh->GetNumElements(), p_diff_type);

    ImmersedBoundaryCellPopulation<2> cell_population(*p_mesh, cells);
    cell_population.SetIfPopulationHasActiveSources(false);

    OffLatticeSimulation<2> simulator(cell_population);
    simulator.SetNumericalMethod(boost::make_shared<ForwardEulerNumericalMethod<2,2> >());
    simulator.GetNumericalMethod()->SetUseUpdateNodeLocation(true);

    // Add main immersed boundary simulation modifier
    MAKE_PTR(ImmersedBoundarySimulationModifier<2>, p_main_modifier);
    simulator.AddSimulationModifier(p_main_modifier);

    // Add force laws
    MAKE_PTR(ImmersedBoundaryLinearMembraneForce<2>, p_boundary_force);
    p_main_modifier->AddImmersedBoundaryForce(p_boundary_force);
    p_boundary_force->SetElementSpringConst(1e7);
    p_boundary_force->SetElementRestLength(0.5);

    // Create and set an output directory that is different for each simulation
    std::stringstream output_directory;
    output_directory << "convergence/num_nodes/sim/" << simulationId;
    simulator.SetOutputDirectory(output_directory.str());

    double dl_at_start = p_mesh->GetAverageNodeSpacingOfElement(0);

    double end_time = 10.0;
    double dt = 0.01;

    // Set simulation properties and solve
    simulator.SetDt(dt);
    simulator.SetSamplingTimestepMultiple(1);
    simulator.SetEndTime(end_time);
    simulator.Solve();

    // Prepare results file to record simulation statistics
    OutputFileHandler results_handler(output_directory.str(), false);
    out_stream results_file = results_handler.OpenOutputFile("results.dat");

    double esf_at_end = p_mesh->GetElongationShapeFactorOfElement(0);

    // Output summary statistics to results file.  lexical_cast is a convenient way to output doubles at max precision.
    (*results_file) << simulationId << ","
                    << boost::lexical_cast<std::string>(dl_at_start) << ","
                    << boost::lexical_cast<std::string>(esf_at_end);

    // Tidy up
    results_file->close();
}
