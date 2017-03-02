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

#include <cxxtest/TestSuite.h>

#include "CellId.hpp"
#include "CellsGenerator.hpp"
#include "CheckpointArchiveTypes.hpp"
#include "ExecutableSupport.hpp"
#include "ExponentialG1GenerationalCellCycleModel.hpp"
#include "ImmersedBoundaryBoundaryCellWriter.hpp"
#include "ImmersedBoundaryHoneycombMeshGenerator.hpp"
#include "ImmersedBoundaryLinearInteractionForce.hpp"
#include "ImmersedBoundaryLinearMembraneForce.hpp"
#include "ImmersedBoundaryMesh.hpp"
#include "ImmersedBoundaryNeighbourNumberWriter.hpp"
#include "ImmersedBoundarySimulationModifier.hpp"
#include "NormallyDistributedTargetAreaModifier.hpp"
#include "OffLatticeSimulation.hpp"
#include "SimpleTargetAreaModifier.hpp"
#include "SmartPointers.hpp"
#include "TransitCellProliferativeType.hpp"

#include <boost/make_shared.hpp>
#include "ForwardEulerNumericalMethod.hpp"

// Program option includes for handling command line arguments
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>

/*
 * Prototype functions
 */
void SetupSingletons(unsigned randomSeed);
void DestroySingletons();
void SetupAndRunSimulation(std::string idString, double corRestLength, double corSpringConst, double traRestLength,
                           double traSpringConst, double interactionDist, unsigned numTimeSteps,
                           bool useNormallyDistAreaModifier);
void OutputToConsole(std::string idString, std::string leading);

int main(int argc, char *argv[])
{
    // This sets up PETSc and prints out copyright information, etc.
    ExecutableSupport::StartupWithoutShowingCopyright(&argc, &argv);

    // Define command line options
    boost::program_options::options_description general_options("This is a Chaste Immersed Boundary executable.\n");
    general_options.add_options()
                    ("help", "produce help message")
                    ("ID", boost::program_options::value<unsigned>()->default_value(0),"ID string for the simulation")
                    ("CRL", boost::program_options::value<double>()->default_value(0.0),"Cortical rest length")
                    ("CSC", boost::program_options::value<double>()->default_value(0.0),"Cortical spring constant")
                    ("TRL", boost::program_options::value<double>()->default_value(0.0),"Transmembrane rest length")
                    ("TSC", boost::program_options::value<double>()->default_value(0.0),"Transmembrane spring constant")
                    ("DI", boost::program_options::value<double>()->default_value(0.0),"Interaction distance for cell-cell forces")
                    ("TS", boost::program_options::value<unsigned>()->default_value(1000u),"Number of time steps")
                    ("NDA", boost::program_options::value<bool>()->default_value(false),"Whether to use normally distributed target areas");

    // Define parse command line into variables_map
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
    unsigned sim_id = variables_map["ID"].as<unsigned>();
    double cor_rest_length = variables_map["CRL"].as<double>();
    double cor_spring_const = variables_map["CSC"].as<double>();
    double tra_rest_length = variables_map["TRL"].as<double>();
    double tra_spring_const = variables_map["TSC"].as<double>();
    double interaction_dist = variables_map["DI"].as<double>();
    unsigned num_time_steps = variables_map["TS"].as<unsigned>();
    bool useNormallyDistAreaModifier = variables_map["NDA"].as<bool>();

    std::string id_string = boost::lexical_cast<std::string>(sim_id);

    OutputToConsole(id_string, "Started");
    SetupSingletons(sim_id);
    SetupAndRunSimulation(id_string, cor_rest_length, cor_spring_const, tra_rest_length, tra_spring_const,
                          interaction_dist, num_time_steps, useNormallyDistAreaModifier);
    DestroySingletons();
    OutputToConsole(id_string, "Completed");
}

void SetupSingletons(unsigned randomSeed)
{
    // Set up what the test suite would do
    SimulationTime::Instance()->SetStartTime(0.0);

    // Reseed with 0 for same random numbers each time, or time(NULL) or simulation_id to change each realisation
    RandomNumberGenerator::Instance()->Reseed(randomSeed);
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

void OutputToConsole(std::string idString, std::string leading)
{
    // Compose the message
    std::stringstream message;
    message << leading << " simulation with ID string " << idString << std::endl;

    // Send it to the console
    std::cout << message.str() << std::flush;
}

void SetupAndRunSimulation(std::string idString, double corRestLength, double corSpringConst, double traRestLength,
                           double traSpringConst, double interactionDist, unsigned numTimeSteps,
                           bool useNormallyDistAreaModifier)
{
    /*
     * 1: Num cells x
     * 2: Num cells y
     * 3: Num nodes per edge
     * 4: Proportional dist between cells
     * 5: Padding in x and y
     */
    ImmersedBoundaryHoneycombMeshGenerator gen(3, 3, 8, 0.025, 0.45);
    ImmersedBoundaryMesh<2, 2>* p_mesh = gen.GetMesh();
    p_mesh->SetNumGridPtsXAndY(256);

    std::vector<CellPtr> cells;
    MAKE_PTR(TransitCellProliferativeType, p_cell_type);
    CellsGenerator<ExponentialG1GenerationalCellCycleModel, 2> cells_generator;
    cells_generator.GenerateBasicRandom(cells, p_mesh->GetNumElements(), p_cell_type);

    ImmersedBoundaryCellPopulation<2> cell_population(*p_mesh, cells);
    cell_population.SetIfPopulationHasActiveSources(true);
    cell_population.AddCellWriter<ImmersedBoundaryBoundaryCellWriter>();
    cell_population.AddCellWriter<ImmersedBoundaryNeighbourNumberWriter>();

    cell_population.SetInteractionDistance(cell_population.GetInteractionDistance() * interactionDist);

    // Loop over all cells
    for (AbstractCellPopulation<2>::Iterator cell_it = cell_population.Begin();
         cell_it != cell_population.End();
         ++cell_it)
    {
        dynamic_cast<ExponentialG1GenerationalCellCycleModel*>(cell_it->GetCellCycleModel())->SetRate(0.1);
        dynamic_cast<ExponentialG1GenerationalCellCycleModel*>(cell_it->GetCellCycleModel())->SetMaxTransitGenerations(5);
        cell_it->GetCellCycleModel()->Initialise();
    }

    OffLatticeSimulation<2> simulator(cell_population);
    simulator.SetNumericalMethod(boost::make_shared<ForwardEulerNumericalMethod<2,2> >());
    simulator.GetNumericalMethod()->SetUseUpdateNodeLocation(true);

    // Add main immersed boundary simulation modifier
    MAKE_PTR(ImmersedBoundarySimulationModifier<2>, p_main_modifier);
    simulator.AddSimulationModifier(p_main_modifier);

    if (useNormallyDistAreaModifier)
    {
        MAKE_PTR(NormallyDistributedTargetAreaModifier<2>, p_area_modifier);
        simulator.AddSimulationModifier(p_area_modifier);
        p_area_modifier->SetReferenceTargetArea(p_mesh->GetVolumeOfElement(0));
    }
    else
    {
        MAKE_PTR(SimpleTargetAreaModifier<2>, p_area_modifier);
        simulator.AddSimulationModifier(p_area_modifier);
        p_area_modifier->SetReferenceTargetArea(p_mesh->GetVolumeOfElement(0));
    }

    // Add force law
    MAKE_PTR(ImmersedBoundaryLinearMembraneForce<2>, p_boundary_force);
    p_main_modifier->AddImmersedBoundaryForce(p_boundary_force);
    p_boundary_force->SetElementRestLength(corRestLength);
    p_boundary_force->SetElementSpringConst(corSpringConst);

    // Add force law
    MAKE_PTR(ImmersedBoundaryLinearInteractionForce<2>, p_cell_cell_force);
    p_main_modifier->AddImmersedBoundaryForce(p_cell_cell_force);
    p_cell_cell_force->SetRestLength(traRestLength);
    p_cell_cell_force->SetSpringConst(traSpringConst);

    std::string output_directory = "ib_numerics_paper/Exe_GrowingPopulationOfCells/sim/" + idString;

    // Set simulation properties
    double dt = 0.01;
    simulator.SetOutputDirectory(output_directory);
    simulator.SetDt(dt);
    simulator.SetSamplingTimestepMultiple(100);
    simulator.SetEndTime(dt * numTimeSteps);

    simulator.Solve();

    std::vector<unsigned> polygon_dist = p_mesh->GetPolygonDistribution();

    OutputFileHandler results_handler(output_directory, false);
    out_stream results_file = results_handler.OpenOutputFile("results.csv");

    // Output summary statistics to results file
    (*results_file) << "id,0-gon,1-gon,2-gon,3-gon,4-gon,5-gon,6-gon,7-gon,8-gon,9-gon,10+gon" << std::endl;

    (*results_file) << idString << ","
                    << boost::lexical_cast<std::string>(polygon_dist[0]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[1]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[2]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[3]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[4]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[5]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[6]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[7]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[8]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[9]) << ","
                    << boost::lexical_cast<std::string>(polygon_dist[10]);

    // Tidy up
    results_file->close();
}
