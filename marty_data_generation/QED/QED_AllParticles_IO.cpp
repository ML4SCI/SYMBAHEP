// g++ -std=c++23 -c QED_AllParticles_IO.cpp -o QED_AllParticles_IO.o 
// g++ -std=c++23 -o QED_AllParticles_IO.x QED_AllParticles_IO.o -lmarty

#include <marty.h>
#include <cxxopts.hpp>
#include <map>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <stdexcept>
// #include <feynOptions.h>
using namespace csl;
using namespace mty;
using namespace std;

using std::cout; using std::cin;
using std::endl; using std::string;
using std::map; using std::copy;


/*
This code utilizes Marco Knipfer's old code for MARTY dataset generation as a base.
Thus, some of the comments have been modified to facilitate my own understanding of MARTY and c++.

The goal of this program is to investigate how to generate Feynman Diagram output in a way that is easier for AMFlow to process it.
The key things AMFlow needs are:

(1) Loop momenta

    mty::FeynmanDiagram objects have a method getNLoops() - returns number of loops in the diagram.
                                            getParticles() - returns set of particles corresponding to a given type - ext/int/loop
                                            Should be able to get loop particle masses if needed
    FeynmanDiagram objects also have a getExpression method - returns expression of FD as a reference; maybe useful for multiple reasons
    
(2) External momenta                

    mty::Kinematics objects have getMomenta()/getOrderedMomenta(), returns external momenta.
    That said, you could also just count the number of external legs and simply set up p1, p2, ..., pn.

(3) Momentum conservation rules     

    For now, just assume p4 -> -p1 -p2 -p3 for all 2-to-2 processes

(4) Replacement rules for momenta (e.g. p^2 -> msq)                                 

    straightfwd, p1^2->0, (p1+p2)^2 is probably okay for now

(5) PROPAGATORS (inc. numerators)                                 

    mty::QuantumField objects have a getPropatagor() method, requires fields and specific momentum as args.
    From the mty::wick::nodes, which have both fields and partner(s),  maybe it's possible to get correct expressions for internal
    propagators?

    Note 10/18/23 - Propagators are propagators in the standard QFT sense, i.e., not simply 1/(p^2 + m^2) type expressions. 
                    They contain the usual tensor structures in the numerators. May need to simplify these expressions
                    for AMFlow as well, since it expects propagators that are/can be used to re-write inner products in the numerator
                    in terms of (p^2 + m^2)-like expressions.

    Note 11/3/23 -  Propagator objects have methods for printing out the propagator expression directly.
                    This code goes from insertion -> amplitudes. Getting diagram/propagator info from the amplitudes is tough
                    since it seems like the Feynman diagram expressions are already abstracted/simplified to a degree where things
                    like explicit propagators/momenta dependence is not present.
                    Must figure out how to obtain diagrams before the amplitude is constructed, if at all possible.

(6) Numerical values of mandelstam vars, msq, etc. - where to evaluate the integral 
	numerically                                                                     

    for now it doesn't matter, since we want symbolic results in the end.

(7) INDICES - which propagators are raised to how many powers. How to get this?     

    Extract from diagrams? QuantumFields?
    Maybe MARTY can count the number of internal propagators. For starters, just say 1 of each propagator, then devise a way
    to count unique instances of propagators and set up the appropriate list for AMFlow.



Gregoire suggests to work directly with Amplitudes and FeynmanDiagram objects as the main object.
Try to extract momenta/propagators from these objects directly. 
We can get kinematics - i.e., explicit momenta and related stuff, via get.Kinematics
Amplitude.getKinematics() - returns kinematics object.
    These objects let you get insertions, momenta, ordered/squared momenta
    mty::Kinematics.getMomenta - returns a vector of momenta tensors, each corresponding to an external momentum
We can extract FeynmanDiagrams from Amplitudes.
FeynmanDiagram.getNLoops() - From this number we can very easily get number of internal momenta.
    N loops -> k1, k2, ..., KN
    However, for internal propagators we still need masses, unless we're working in the massless limit.
    diagram.getParticles() should give a QuantumFieldParent-like object, from which we can extract masses/

if possible, we want to try and export momenta and propagators in an easy to access/process form
For example, maybe
list = {{k1, k2}, {p1, p2, p3, p4}, {k1^2, (k1+p1)^2, k2^2, (k2+p2)^2 - msq}}
So that list[0] gives loop momenta, list[1] gives ext, list[2] gives full list of propagators...
Essentially, we want it so that we can just say something like 

cout << "AMFlowInfo[\"Family\"] = " << some string, probably just a number << endl;
cout << "AMFlowInfo[\"Loop\"] = " << list[0] << endl;
cout << "AMFlowInfo[\"Leg\"] = " << list[1] << endl;
cout << "AMFlowInfo[\"Conservation\"] = " << something << endl;
cout << "AMFlowInfo[\"Replacement\"] = " << something << endl;
cout << "AMFlowInfo[\"Propagator\"] = " << list[2] << endl;
cout << "AMFlowInfo[\"Numeric\"] = " << {s -> 100, t -> -10/3, msq -> 1} << endl;
cout << "AMFlowInfo[\"NThread\"] = " << "4" << endl;

We can just output everything into one mmca block and run it straightforwardly. So it should be possible to setup files here
and write directly to .wl files.

*/













vector<string> split(const string& i_str, const string& i_delim)
    // from https://stackoverflow.com/a/57346888 
    // split a string at each appearance of i_delim
    // like python `string.split()`
{
    vector<string> result;						//This is the output vector - a vector of strings.
    
    //size_t is the unsigned integer type of the result of 'sizeof' operator and 'alignof' 
    //operator. It can store the max size of a theoretically possible object of any type.
    //Commonly used for array indexing and loop counting. 
    size_t found = i_str.find(i_delim);			//Find the first instance of i_delim
    size_t startIndex = 0;						//Set the starting string index = 0

    while(found != string::npos)				//While i_delim exists in i_str (found != -1)
    {
        result.emplace_back(string(i_str.begin()+startIndex, i_str.begin()+found));	//string(a, b) gets string between indices a and b of i_str.
        startIndex = found + i_delim.size();										//Update startIndex to be after first instance of i_delim.
        found = i_str.find(i_delim, startIndex);									//Update found to be next instance of i_delim.
    }		
    if(startIndex != i_str.size())													//If there are no more delimiters in i_str, but 
        result.emplace_back(string(i_str.begin()+startIndex, i_str.end()));			//startIndex isn't at the end of the string,
    return result;      															//append everything from last delimiter to string end.
}


bool isInternalVertex(csl::Tensor const &X)
//As name suggests, checks if a csl tensor object corresponds to  an internal vertex of a Feynman diagram.
//X is a reference to the input tensor, not the tensor itself.
{
    const auto &name = X->getName();			//dereferences member getName() of input tensor reference X, sets it to reference name
    return !name.empty() && name[0] == 'V';		//If name not empty and the first character is 'V', this is an internal vertex.
}


struct SimpleField
// Simple enough, this defines a structure representing a simple field.
// Has name, vertex string attributes and two boolean attribues, p and s.
{
    std::string name;
    std::string vertex;
    bool p;
    bool s;
};

SimpleField convertField(mty::QuantumField const &field)
// Type is SimpleField, function is convertField.
// Takes input field (a mty::QuantumField object) and returns a corresponding SimpleField object
// To test if this works, we need to create a mty::QuantumField object somehow.
// This function is used in getDiagramConnections() function.
{
    std::string name = field.getName();
    std::string vertexName = field.getPoint()->getName();
    bool p = field.isParticle();
    bool s = field.isOnShell();
    return {name, vertexName, p, s};
}




// Connection objects represents external fields that connect
// to the same internal vertex
struct Connection {
    std::string vertex;
    std::vector<SimpleField> externalFields; // List of external fields connected
};


// Returns the list (a vector object) of connections corresponding to the diagram's topology
// mty:wick:Node - class representing a contractible mty::QuantumField in the context of Wick contraction.
// std::shared_ptr - smart pointer retaining shared ownership of an obj through a pointer.
// 
// Takes in a reference to nodes - a vector of shared pointers for mty::wick::Node objects 
// (Shared pointers b/c the same node can be connected to multiple vertices?)
//
// For AMFlow, we may eventually need to know the topology and how any given connections are actually formed/represented.
// By selecting the appropriate connections and the type of particles they represent, maybe we can generate or extract
// the corresponding propagators/momenta.
//
// Note that the QuantumField objects have a getPropagator(const QuantumField &other, csl::Tensor &vertex) method, returning a csl::Expr.
// It returns the propagator with another QuantumField. 
// other - the other field involved
// vertex - momentum integrated in the propagator
std::vector<Connection> getDiagramConnections(std::vector<std::shared_ptr<mty::wick::Node>> const &nodes)
{
    std::unordered_map<std::string, Connection> connections;		// connections - keys are strings, values are connection structs.
    for (const auto &node : nodes) {								// iterate through references to nodes in input nodes
        auto field = *node->field;									// get field represented by node in the graph
        auto partner = *node->partner.lock()->field;				// get contracted partner field, save as "partner" 
        bool fieldInternal = isInternalVertex(field.getPoint());	// Check if current/partner field is on internal vertex
        bool partnerInternal = isInternalVertex(partner.getPoint());// what does getPoint actually do though???
        if (partnerInternal)										// if partner is internal
        {
            csl::Tensor vertex;										// create csl::Tensor object
            if (fieldInternal)										// If field also internal
                vertex = field.getPoint();							// get corresponding internal vertex for field
            else
                vertex = partner.getPoint();						// get corresponding internal vertex for contracted partner
            auto pos = connections.find(vertex->getName());			// look for specific vertex in connections using getName()
            if (pos == connections.end()) {
            	// if not in connections, Add new Connection {vertex name, converted field} to end of connections with the key vertex->getName() 
                connections[vertex->getName()] = Connection{vertex->getName(), {convertField(field)}}; 
            }
            else {
            	// if in connections, Add converted field to externalFields vector attr of connections[vertex->getName()]
                connections[vertex->getName()].externalFields.push_back(convertField(field));
            }
        }
        if (fieldInternal)											// if field itself is internal
        {
            csl::Tensor vertex;										// create csl::Tensor object
            if (partnerInternal)									// If partner also internal
                vertex = partner.getPoint();						// get corresponding internal vertex for partner
            else
                vertex = field.getPoint();							// get corresponding internal vertex for field
            auto pos = connections.find(vertex->getName());			// look for specific vertex in connections using getName()
            if (pos == connections.end()) {
            	// if not in connections, Add new Connection{vertex name, converted partner} to end of connections with the key vertex->getName() 
                connections[vertex->getName()] = Connection{vertex->getName(), {convertField(partner)}};
            }
            else {
            	// if in connections, Add converted partner to externalFields vector attr of connections[vertex->getName()]
                connections[vertex->getName()].externalFields.push_back(convertField(partner));
            }
        }
    }
    std::vector<Connection> res;					// Vector of connection objects
    res.reserve(connections.size());				// Make it the same size as connections
    for (const auto &el : connections) {			// for each element in connections, i.e. each 
        res.push_back(std::move(el.second));		// move 2nd element of el (i.e., the connection itself) from to res
    }

    return res;
}

/*	So in the end, the above function simply returns a vector {cnx1, cnx2, ...} of connection structures.
	Each connection structure has its own Vertex, and a list of External fields.
	But this doesn't tell us anything about which fields on a vertex are connected to fields on other vertices, right?
	The fact that nodes can have partners already tells us about which fields are paired/contracted.
*/




/*
// Determines if a list of connections representing a diagram
// is a s-channel. True if external fields on X_1 and X_2 are
// found in the same connection (corresponds to momenta p_1 and p_2)
//
// I don't think this is actually called anywhere else, so I'll comment it out for now.
bool isSChannel(std::vector<Connection> const &topology)
{
    for (const auto &conn : topology) {
        if (conn.externalFields.size() != 2) {
            continue;
        }
        auto nameA = conn.externalFields[0].vertex;
        auto nameB = conn.externalFields[1].vertex;
        if ((nameA == "X_1" && nameB == "X_2")
                || (nameA == "X_2" && nameB == "X_1")) {
            return true;
        }
    }
    return false;
}
*/



// Processes an amplitude, finding the topologies
// and printing the connections.

std::vector<std::vector<Connection>> processAmplitudes(Amplitude const &ampl)
// Takes in a reference to an Amplitude object (mty::Amplitude?) &ampl
// Returns a vector of vectors of connection objects.
{
    // Get diagrams
    std::vector<FeynmanDiagram> const &diagrams = ampl.getDiagrams();	// getDiagrams(from the input amplitude)
    std::vector<std::vector<Connection>> topologies;					// setup a vector of connection vectors - topologies
    topologies.reserve(diagrams.size());								// topologies should be same size as # of diagrams
    for (auto const &diag : diagrams) {									// for each diagram
        //std::cout << "\n************\n";
        //std::cout << "New diagram:\n";
        // Getes the nodes of the diagram
        // (contains edges by looking at the nodes' partners)
        auto nodes = diag.getDiagram()->getNodes();						// Get nodes from the diagram
        auto connections = getDiagramConnections(nodes);				// get connections from diagram nodes
        topologies.push_back(std::move(connections));					// add vector of connections to topologies
    }
    return topologies;													// return full vector topologies
}






void export_diagrams_str(mty::Amplitude ampl, std::ofstream &stream)                    
// Take in an amplitude and an output stream
// Seems to just print the output for a given diagram amplitude- involved particles, on/off shell, vertices
{
    std::vector<std::vector<Connection>> diagramTopologies = processAmplitudes(ampl);   // A vector of vectors of connections - processed diagrams
    for (auto const &diagram : diagramTopologies) {                                     // for each vector in diagramTopo (processed diagram)
        for (const Connection &conn : diagram) {                                        // for each connection in the vector
            stream << "Vertex " << conn.vertex << ":";                                  // Output the vertex
            for (const SimpleField &field : conn.externalFields)                        // for each external field
            {
                // Output particle/antiparticle, on/off shell, Particle/field name, and associated vertex
                // e.g. AntiParticle Onshell b(V_0)
                stream << (field.p ? "Particle" : "AntiParticle") << (field.s ? " OnShell " : " OffShell ") <<field.name << "(" << field.vertex << "),";
            }
            stream << endl;
        }
        stream << "--------------" << endl;
    }
}



void export_feynman_diagrams_str(mty::Amplitude process_ampl, std::ofstream &stream)
// Goal: take in an amplitude/output stream. Output/write FeynmanDiagram expression to output file.
// The output of these FeynmanDiagram expressions is not user friendly. Look for another way.
{
    std::vector<mty::FeynmanDiagram> diagrams = process_ampl.getDiagrams();            // a vector of FeynmanDiagram objects
    // std::vector<csl::Expr> fd_expressions = {};                                     // Empty vector meant to house csl expressions of fds

    for(size_t i=0; i!=diagrams.size(); i++){
        std::vector<mty::FeynmanDiagram> diagram = {diagrams[i]};                      // diagrams[i] is a FD object.
        std::vector<mty::Particle> loopParticles;
        for(size_t j=0; j!=diagram.size(); j++){
            // loopParticles = diagram[j].getParticles(FeynmanDiagram::Loop);
            stream << diagram[j].getExpression() << endl;
            stream << diagram[j].getNLoops() << endl;
            // for(size_t k=0; k!=loopParticles.size(); k++){
            //     stream << loopParticles[k]->getName() << endl;
            //     stream << "--------------" << endl;
            // }
            
            
        }
    }

    // for(auto const &diagram : diagrams){                                         // For each vector of diagrams in diagrams
    //     stream << diagram.getExpression() << endl;                               // Output the expression for the diagram
    // }
    stream << "--------------" << endl;
}




std::vector<csl::Expr> square_amplitude_individually(mty::Amplitude process_ampl, mty::Model& model)
// Takes in a mty::Amplitude and a mty:Model.
// Calculates squared amplitude given an amplitude and a model.
{
    auto opts = process_ampl.getOptions();
    auto kinematics = process_ampl.getKinematics();
    std::vector<mty::FeynmanDiagram> diagrams = process_ampl.getDiagrams();
    std::vector<csl::Expr> squared_ampl_expressions = {};

    for(size_t i=0; i!=diagrams.size(); i++){
        std::vector<mty::FeynmanDiagram> diagram = {diagrams[i]};
        auto ampl = mty::Amplitude(opts, diagram, kinematics);
        auto square = model.computeSquaredAmplitude(ampl);
        auto square_eval = Evaluated(square, eval::abbreviation);
        // auto square_eval = square;
        squared_ampl_expressions.push_back(square_eval);
    }
    
    return squared_ampl_expressions;
};


// string format_momenta(std::vector<std::string> momenta, string separator = ","){
//     // std::string separator = ",";
//     std::string formatted_string="{";
//     for (auto x : momenta) {
//         formatted_string = formatted_string + x + separator;
//     }
//     formatted_string = formatted_string.erase(formatted_string.find_last_of(separator));
//     formatted_string = formatted_string + "}";
//     return formatted_string
// }




string return_loop_momenta(mty::FeynmanDiagram diag, string separator = ",")
//  For a given diagram diag, get number of internal loops and construct a formatted string with the preferred
//  separator/delimiter. Default is ",".
//  e.g. returns the string "{k1, k2, ..., kn}"
//  If there are no loops, returns "{}"
//  This is pretty unrefined, but it works.
{
    std::string formatted_string = "{";
    int loops = diag.getNLoops();
    if (loops==0){
        formatted_string = formatted_string + "}";
        return formatted_string;
    }

    for(int i=0; i!=loops; i++){
        formatted_string = formatted_string + "k" + std::to_string(i+1) + separator;
    }
    formatted_string = formatted_string.erase(formatted_string.find_last_of(separator));
    formatted_string = formatted_string + "}";
    return formatted_string;
}



string expr_to_string(csl::Expr &expr){
    // Print the expr as a string right away so it's parseable with standard methods.
    std::ostringstream a;
    a << expr;
    std::string b = std::move(a).str();
    return b;
}

string extract_loop_int(csl::Expr &fdexpr, string i_delim){
    // (1) Find the string/item "int{"
    // (2) create a size_t variable i to store the first index of "int{"
    // (3) remove the terms/expressions before index of fdexpr
    // (4) return trimmed fdexpression


    std::string fdexpr_string = expr_to_string(fdexpr);
    // if our delimiter is "int{," then i should be the position of "i."
    size_t i = fdexpr_string.find(i_delim);
    // This line should extract the part of the string from the start of "int{" until the end of the expression - i.e., the integral.
    fdexpr_string = fdexpr_string.substr(i);
    return fdexpr_string;
}





std::string get_loop_propagators(mty::FeynmanDiagram const diag, std::ofstream &stream)
// Work-in-progress
// For a given diagram diag, find which particles are in loops and return list of propagators.
// getParticles only lists the unique particles involved in loops, NOT each instance of particles involved in loops.
// so if a loop involves two electrons, it only sees "one" electron, because only one type of particle runs in the loop.
// Workaround - brute force parsing by finding instances of "Prop(" in the string version of the csl::Expr

{
    // std::vector<Particle> loopParticles = diag.getParticles(FeynmanDiagram::Loop);
    // std::vector<Particle> mediParticles = diag.getParticles(FeynmanDiagram::Mediator);
    // int num_prop = loopParticles.size() + mediParticles.size();

    csl::Expr diag_expr = diag.getExpression();
    csl::Expr loop_int = diag_expr->getArgument(Size(diag_expr)-1); //Loopint
    
    std::string integral_str = expr_to_string(loop_int);    // Returns just int{...} onwards as a string
    std::string prop_start = "Prop(";                       // Identifying where in the return string to swap
    std::string prop_end = ")";                             // out the old Prop(...) with the new one.
    size_t found = integral_str.find(prop_start);
    size_t found2 = integral_str.find(prop_end, found+1);
    size_t startIndex = 0;
    
    csl::Expr integrand = loop_int->getArgument(1);        //Integrand everything after int{...}
    
    // This feels like a silly solution, but we need to parse the string until we know that there are no more "Prop()" left.
    // Basically, we need an index to pass to getArgument.
    int  loop_count = 0;

    while(found != string::npos)
    {
        csl::Expr mom = integrand->getArgument(loop_count)->getArgument(0);      // Momentum arg of Prop() - an Expr
        csl::Expr mass = integrand->getArgument(loop_count)->getArgument(1);      // Mass arg of Prop() - an Expr

        // Create new propagator object and evaluate in the appropriate form.
        mty::Propagator temp_prop = Propagator(mom, mass, 0);
        std::optional<csl::Expr> newprop_expr = temp_prop.evaluate(csl::eval::abbreviation);
        std::string newprop_str = expr_to_string(*newprop_expr);

        // Remove residual "reg_prop" from propagator string. Not sure where it comes from, honestly.
        std::string regprop = "reg_prop";
        size_t regprop_ind = newprop_str.find(regprop);
        newprop_str.replace(regprop_ind, regprop.size() + 3, "");       // Replaces "reg_prop + " string with ""

        // Replace old propagator with new one in the loop integral string
        integral_str.replace(found, found2 - found + 1, newprop_str);
                                  
        // Update starting points for string search                                  
        found = integral_str.find(prop_start, startIndex);
        found2 = integral_str.find(prop_end, found+1);
        
        // Increase loop count for getArgument
        loop_count = loop_count+1;
    }


    return integral_str;
}



void get_loop_int_str(mty::Amplitude process_ampl, std::ofstream &stream)
// Goal: take in an amplitude/output stream. Output/write the loop integral string to the output stream.
// TODO: (1) Figure out how to "evaluate" propagators to go from Prop(p_1_\rho1, m) to 1/(p_1_%\rho1 p_1_+%\rho1 - m^2) or something.
{
    std::vector<mty::FeynmanDiagram> diagrams = process_ampl.getDiagrams();            // a vector of FeynmanDiagram objects
    std::vector<csl::Expr> diag_expr = process_ampl.obtainExpressions();               // vector meant to house csl expressions of fds

    for(size_t i=0; i!=diag_expr.size(); i++){
        // Note - a lot of this is leftover from debugging and figuring out how things work.
        // Parse arguments of the diagram expression.
        // It appears that the integral is the last arg/term, containing the remaining propagators/loop-momenta dependent objects

        // Returns the integral term of the overall DF expression, notably without "int{...}".
        // loopint = int{...}... itself IS an expression, but for some reason we can't parse it arg by arg to extract the propagators.
        csl::Expr loopint = diag_expr[i]->getArgument(Size(diag_expr[i])-1);
        // stream <<  "Integral Expr size is :" <<Size(loopint) << endl;     // the integral is an expression of size 1. We go deeper recursively


        csl::Expr integrand = loopint->getArgument(1);                  // Full integrand                                    
        // std::string inttype = typeid(integrand).name();                 // Integrand is ALSO an Expr
        // stream << "Integrand Expr is of size: " << Size(integrand) << endl;  // Integrand has size > 1! Can extract/manipulate terms maybe?

        std::string loopint_str = get_loop_propagators(diagrams[i], stream);    //This function is defined above.
        stream << loopint_str << endl;

    }
}




void get_diagram_exprs(mty::Amplitude process_ampl, std::ofstream &stream)
// 2/23/2024 - For now this is a "test" method for checking out things to manipulate/obtain things for a single diagram.
// The goal is for this to simple access the amplitude, get the wick graph/amplitude expression, and print out the
// expression for the diagram to an output stream.
// 
{
    std::vector<csl::Expr> diag_expr = process_ampl.obtainExpressions();                     // vector of diagrams in amplitude
    std::vector<std::shared_ptr<mty::wick::Graph>> diag_graph = process_ampl.obtainGraphs(); // vec of pointers to mty::wick:Graphs
    std::vector<mty::FeynmanDiagram> diagrams = process_ampl.getDiagrams();
    int ndiag = diagrams.size();                                                    // This gets the TOTAL number of diagrams.
    mty::Kinematics process_kin = process_ampl.getKinematics();                     // Gets Kinematics object of amplitude
    std::vector<csl::Tensor> momenta_p = process_kin.getOrderedMomenta();           // Returns vector of momenta in order (p_1, p_2,...)
    std::vector indices_p = process_kin.getIndices();                               // Get kinematic indices - "1, 2, 3, etc..."

    // 
    // We need a TensorElement to extract the lorentz/gamma index information.
    // TDerivative gives an operator. It needs a name and space. When applying to an Expr, must provide the index/Tensor. NOT a TensorElement.
    // 
    // stream << diag_expr[0] << endl;
    stream << extract_loop_int(diag_expr[0], "int{") << endl;
    // stream << "----------------------------------------------------------" << endl;
}

















void export_amflow_format(mty::Amplitude process_ampl, std::ofstream &stream)
// This function should extract the relevant information from a specific diagram and export it as a .wl file.
// Current implementation is very brute force, and exports everything to a SINGLE .wl file.
// Eventually this will be modified to output a single .wl file for EACH diagram.
// 
// Since we need one of these per diagram amplitude, we can't simply call the function once. Either it should be called
// in a loop over the vector of procesed amplitudes, or take in a vector of processed amplitudes and iterate over those.
// Far future to-do - leverage diagram topology so FI families can be specified and cut down on number of .wl files created.
  // 
// to do - extract loop momenta - DONE.
//         extract external momenta - DONE. 
//         extract/create propagators - WIP
// 
// Note - this goes through all the diagrams, but the number of diagrams per process is wierd somehow. Have to look into it or find
//        an alternative for naming the diagram families correctly.
//
// 11/3/23:
// FIND THE PROPAGATOR CREATION STEP
// THERE'S A PART WHERE MARTY JUST EXPLICITY CONSTRUCTS THE PROPAGATOR(S)
// AND YOU CAN JUST ADD PRINT STATEMENTS WHERE NEEDED
// We still may need to use CSL to contract indices/simplify/find out which momenta inner products already exist so that we can
// supplement/complete the basis for the problem.
// 
// FIs also explicitly accesses the LoopTools package, so that should get loop/ext momenta.
// Look into LoopTools
{   
    std::vector<mty::FeynmanDiagram> diagrams = process_ampl.getDiagrams();
    int ndiag = diagrams.size();                                                    // This gets the TOTAL number of diagrams.
    mty::Kinematics process_kin = process_ampl.getKinematics();                     // Gets Kinematics object of amplitude
    std::vector<csl::Tensor> momenta_p = process_kin.getOrderedMomenta();           // Returns vector of momenta in order (p_1, p_2,...)
    std::vector<std::string> names_p;                                               // Need to be processed to "{p1, p2,...}""
    std::string name_p;
    std::vector<Particle> loopParticles;

    for(size_t i = 0; i != momenta_p.size(); i ++){                                 // For each momentum tensor in momenta_p
        // Interestingly, csl::Tensor objects inherit from a std::shared_ptr<TensorParent>
        // So they don't have methods available to TensorParent.
        name_p = momenta_p[i]->getName();                               // get the name of the momentum (string) - doesn't throw an error!
        std::vector<std::string> temp_name_p = split(name_p, "_");                  // split by _            
        name_p = temp_name_p[0] + temp_name_p[1];                                   // go from p_1 -> p1
        names_p.emplace_back(name_p);                                               // Append name to names_p (vector of strings)
    }

    std::string separator = ",";                                                    // This processes {p_1, p_2,...}} to "{p1, p2,...pn}"
    std::string names_p_string="{";                                                 // Find a more elegant way to do thhis for ext and
    for (auto x : names_p) {                                                        // int loop momenta!!!
        names_p_string = names_p_string + x + separator;
    }
    names_p_string = names_p_string.erase(names_p_string.find_last_of(","));
    names_p_string = names_p_string + "}";

    std::vector<string> particleNames;

    for(int j=0; j!=ndiag; j++){
        stream << "Current diagram number in process_ampl is " << j << " out of ndiag = " << ndiag << "." << endl;
        // particleNames = get_loop_propagators(diagrams[j]);               // This function is obsolete
        // stream << "Loop Particles: " << particleNames << endl;
        stream << "-------------------------------------------------------------------------------------------------------------------------" << endl;
        stream << "current = If[$FrontEnd===Null,$InputFileName,NotebookFileName[]]//DirectoryName;" << endl;
        stream << "Get[FileNameJoin[{current, \"..\", \"..\", \"..\",\"software\",\"amflow\", \"AMFlow.m\"}]];" << endl;
        stream << "SetReductionOptions[\"IBPReducer\" -> \"FIRE+LiteRed\"];" << endl;
        stream << endl;
        stream << "AMFlowInfo[\"Family\"] = " << j << ";"<< endl;
        std::string extKFormatted = return_loop_momenta(diagrams[j]);                   // Get formatted loop momenta list. Done on a 
        stream << "AMFlowInfo[\"Loop\"] = " << extKFormatted << endl;                   // diag by diag basis. can't combine with p_i's.
        stream << "AMFlowInfo[\"Leg\"] = " << names_p_string << ';' << endl;
        stream << "AMFlowInfo[\"Conservation\"] = {p4 -> -p1-p2-p3};" << endl;          // Okay for 2-to-2, probably. Automate for n-to-m
        stream << "AMFlowInfo[\"Replacement\"] = {p1^2 -> 0, p2^2 -> 0, p3^2 -> 0, p3^2 -> 0,  p4^2 -> 0, (p1 + p2)^2 -> s, (p1 + p3)^2 -> t};" << endl;
        stream << "AMFlowInfo[\"Propagator\"] = " << "temp" << ";" << endl;
        stream << "AMFlowInfo[\"Numeric\"] = {s -> 100, t -> -1, msq -> 1};" << endl;    // Okay for now, what if multiple masses?
        stream << "AMFlowInfo[\"Nthread\"] = 4;" << endl;                               // 4 threads is okay
        stream << endl;
        stream << "integrals = " << "temp" << ";" << endl;
        stream << "precision = 3;" << endl;
        stream << "epsorder = 2;" << endl;
        stream << "sol1 = SolveIntegrals[integrals, precision, epsorder];" << endl;
        stream << "Put[sol1, FileNameJoin[{current, \"sol1\"}]];" << endl;
        stream << "-------------------------------------------------------------------------------------------------------------------------" << endl;
    }

}









mty::Insertion get_insertion(string name){
    // This returns the correct field insertion for the particle passed as name.
    // Somewhere, this should be called when we're getting/calculating amplitudes
    // ToDo:
    // Write function nicer with `split`!
    auto name_split = split(name, "_");
    if (name_split[0] == "OffShell")
        {   // If first substring is "OffShell", pick off "OffShell" and re-do get_insertion for just the particle name.
            // Then wrap the whole insertion in OffShell(), so it becomes OffShell(*insertion*).
        auto name_new = name.substr(9);
        // cout << "OffShell: " << name_new << endl;
        auto ret = OffShell(get_insertion(name_new));
        // cout << "isOnShell: " << ret.isOnShell() << endl;
        return ret;
    }

    // electron
    if ((name == "in_normal_electron") || (name == "in_electron"))
        return Incoming("e");
    else if (name == "in_anti_electron")
        return Incoming(AntiPart("e"));
    else if ((name == "out_electron") || (name == "out_normal_electron"))
        return Outgoing("e");
    else if (name == "out_anti_electron")
        return Outgoing(AntiPart("e"));

    // muon
    if ((name == "in_normal_muon") || (name == "in_muon"))
        return Incoming("mu");
    else if (name == "in_anti_muon")
        return Incoming(AntiPart("mu"));
    else if ((name == "out_muon") || (name == "out_normal_muon"))
        return Outgoing("mu");
    else if (name == "out_anti_muon")
        return Outgoing(AntiPart("mu"));

    // tau
    if ((name == "in_normal_tau") || (name == "in_tau"))
        return Incoming("t");
    else if (name == "in_anti_tau")
        return Incoming(AntiPart("t"));
    else if ((name == "out_tau") || (name == "out_normal_tau"))
        return Outgoing("t");
    else if (name == "out_anti_tau")
        return Outgoing(AntiPart("t"));

    // up
    if ((name == "in_normal_up") || (name == "in_up"))
        return Incoming("u");
    else if (name == "in_anti_up")
        return Incoming(AntiPart("u"));
    else if ((name == "out_up") || (name == "out_normal_up"))
        return Outgoing("u");
    else if (name == "out_anti_up")
        return Outgoing(AntiPart("u"));

    // down
    if ((name == "in_normal_down") || (name == "in_down"))
        return Incoming("d");
    else if (name == "in_anti_down")
        return Incoming(AntiPart("d"));
    else if ((name == "out_down") || (name == "out_normal_down"))
        return Outgoing("d");
    else if (name == "out_anti_down")
        return Outgoing(AntiPart("d"));

    // strange
    if ((name == "in_normal_strange") || (name == "in_strange"))
        return Incoming("s");
    else if (name == "in_anti_strange")
        return Incoming(AntiPart("s"));
    else if ((name == "out_strange") || (name == "out_normal_strange"))
        return Outgoing("s");
    else if (name == "out_anti_strange")
        return Outgoing(AntiPart("s"));

    // charm
    if ((name == "in_normal_charm") || (name == "in_charm"))
        return Incoming("c");
    else if (name == "in_anti_charm")
        return Incoming(AntiPart("c"));
    else if ((name == "out_charm") || (name == "out_normal_charm"))
        return Outgoing("c");
    else if (name == "out_anti_charm")
        return Outgoing(AntiPart("c"));

    // bottom
    if ((name == "in_normal_bottom") || (name == "in_bottom"))
        return Incoming("b");
    else if (name == "in_anti_bottom")
        return Incoming(AntiPart("b"));
    else if ((name == "out_bottom") || (name == "out_normal_bottom"))
        return Outgoing("b");
    else if (name == "out_anti_bottom")
        return Outgoing(AntiPart("b"));

    // top
    if ((name == "in_normal_top") || (name == "in_top"))
        return Incoming("tt");
    else if (name == "in_anti_top")
        return Incoming(AntiPart("tt"));
    else if ((name == "out_top") || (name == "out_normal_top"))
        return Outgoing("tt");
    else if (name == "out_anti_top")
        return Outgoing(AntiPart("tt"));

    else if ((name == "in_photon") || (name == "in_normal_photon"))
        return Incoming("A");
    else if ((name == "out_photon") || (name == "out_normal_photon"))
        return Outgoing("A");
    else {
        cout << "particle " << name << "not found" << endl;
        throw std::invalid_argument("received unknown particle "+name);
        // return Incoming("e");
    }
}



void print_help_func(){
    cout << "help" << endl;

    cout << "--help: print this help" << endl;
    cout << "--particles=in_electron,in_anti_electron,out_photon: insertion arbitrary amount of insertion particles, separated by comma, no space." << endl;
    cout << "--famplitudes: file where the amplitudes should be saved, default: out/ampl.txt" << endl;
    cout << "--famplitudes_raw: file where the raw amplitudes should be saved, default: out/ampl_raw.txt" << endl;
    cout << "--fsqamplitudes: file where the squared amplitudes should be saved, default: out/ampl_sq.txt" << endl;
    cout << "--fsqamplitudes_raw: file where the raw squared amplitudes should be saved, default: out/ampl_sq_raw.txt" << endl;
    cout << "--fdiagrams_str: file where the diagrams strings should be saved, default: out/ampl_sq_raw.txt" << endl;
    cout << "--fdexpr: file where the Feynman Diagram csl expressions should be saved, default: out/fdexpr.txt" << endl;
    cout << "--loopint_str: file where the amflow mathematica script should be saved, default: out/loopint_str.txt" << endl;
    cout << "--diagrams: If diagrams should be shown, default: false" << endl;
    cout << "--append: If files should be appended or replaced" << endl;
}












/*THIS IS A TESTING main() FOR DEBUGGING AND MAKING SURE ANY ADJUSTED FUNCTIONS WORK AS INTENDED*/
int main(int argc, char const *argv[])
{
	/*
	string testString = "This, is, a, comma, delimited, object, for, testing, purposes.";
	auto split_str = split(testString, ", ");
	for(int i = 0; i != split_str.size(); i++){
		cout << split_str[i] << "\n";
	}
	*/

/*
	std::vector<int> origInt = {10, 20, 30, 40, 50};
	cout << origInt << endl; 
	std::vector<int> tempInt;					
    tempInt.reserve(origInt.size());				
    for (const auto &el : origInt) {			
        tempInt.push_back(std::move(el.second));		
    }
    cout << origInt << endl;
    cout << tempInt << endl;
*/


    auto export_insertions = false;  // useless anyways ... it's already in the file name

    /*
    Main part of the program
    Specify heop and options for running the file/data generation
    */
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
      ("h,help", "Print help", cxxopts::value<bool>()->default_value("false")) // a bool parameter
      ("f,fdexpr", "File name for (f)eynman diagram expressions", cxxopts::value<std::string>()->default_value("out/fdexpr.txt"))
      ("m,loopint_str", "File name for output loop integral files", cxxopts::value<std::string>()->default_value("out/loopint.txt"))
      ("a,famplitudes", "File name for amplitudes", cxxopts::value<std::string>()->default_value("out/ampl.txt"))
      ("s,fsqamplitudes", "File name for squared amplitudes", cxxopts::value<std::string>()->default_value("out/ampl_sq.txt"))
      ("r,fsqamplitudes_raw", "File name for raw squared amplitudes", cxxopts::value<std::string>()->default_value("out/ampl_sq_raw.txt"))
      ("t,famplitudes_raw", "File name for raw amplitudes", cxxopts::value<std::string>()->default_value("out/ampl_raw.txt"))
      ("i,finsertions", "File name for insertions. This is a remnant and will not do anything any more!", cxxopts::value<std::string>()->default_value("out/insertions.txt"))
      ("d,diagrams", "Show diagrams", cxxopts::value<bool>()->default_value("false"))
      ("b,fdiagrams_str", "File name for insertions", cxxopts::value<std::string>()->default_value("out/diagrams.txt"))
      ("p,particles", "Insertion particles", cxxopts::value<std::vector<std::string>>())
      ("e,append", "append to files (extend)", cxxopts::value<bool>()->default_value("false"))
      ;

    auto opts = options.parse(argc, argv);
    auto print_help = opts["help"].as<bool>();
    auto print_diagrams = opts["diagrams"].as<bool>();
    auto append_files = opts["append"].as<bool>();
    auto fdexpr_file = opts["fdexpr"].as<std::string>();
    auto loopint_file = opts["loopint_str"].as<std::string>();
    auto particles_strings = opts["particles"].as<std::vector<std::string>>();
    auto amplitudes_file = opts["famplitudes"].as<std::string>();
    auto sqamplitudes_file = opts["fsqamplitudes"].as<std::string>();
    auto sqamplitudes_raw_file = opts["fsqamplitudes_raw"].as<std::string>();
    auto amplitudes_raw_file = opts["famplitudes_raw"].as<std::string>();
    auto diagrams_file = opts["fdiagrams_str"].as<std::string>();

    if (print_help){
        print_help_func();
        return 0;
    };
    cout << "Will export raw amplitudes to " << amplitudes_raw_file << endl;
    cout << "Will export amplitudes to " << amplitudes_file << endl;
    cout << "Will export squared amplitudes to " << sqamplitudes_file << endl;
    cout << "Will export raw squared amplitudes to " << sqamplitudes_raw_file << endl;
    cout << "Will export diagrams to " << diagrams_file << endl;
    cout << "Will export Feynman diagram expressions to " << fdexpr_file << endl;
    cout << "Will export loop integral strings to " << loopint_file << endl;
    if (append_files)
        cout << "Files will be appended if they exist." << endl;
    else
        cout << "Files will be overwritten if they exist." << endl;


    /*
    Setting up QED model. Defining gauge group, particles content, group reps,
    */
    Model QED;
    Expr psi = constant_s("e");
    QED.addGaugedGroup(group::Type::U1, "em", psi);

    QED.init();

    Particle e = diracfermion_s("e", QED);
    Particle mu = diracfermion_s("mu", QED);
    Particle t = diracfermion_s("t", QED);
    Particle u = diracfermion_s("u", QED);
    Particle d = diracfermion_s("d", QED);
    Particle s = diracfermion_s("s", QED);
    Particle tt = diracfermion_s("tt", QED);
    Particle c = diracfermion_s("c", QED);
    Particle b = diracfermion_s("b", QED);

    auto m_e = constant_s("m_e");
    auto m_mu = constant_s("m_mu");
    auto m_t = constant_s("m_t");
    auto m_u = constant_s("m_u");
    auto m_d = constant_s("m_d");
    auto m_s = constant_s("m_s");
    auto m_tt = constant_s("m_tt");
    auto m_c = constant_s("m_c");
    auto m_b = constant_s("m_b");

    e->setGroupRep("em", -1);
    mu->setGroupRep("em", -1);
    t->setGroupRep("em", -1);
    u->setGroupRep("em", {2,3});
    d->setGroupRep("em", {-1,3});
    s->setGroupRep("em", {-1,3});
    tt->setGroupRep("em", {2,3});
    c->setGroupRep("em", {2,3});
    b->setGroupRep("em", {-1,3});

    e->setMass(m_e);
    mu->setMass(m_mu);
    t->setMass(m_t);
    u->setMass(m_u);
    d->setMass(m_d);
    s->setMass(m_s);
    tt->setMass(m_tt);
    c->setMass(m_c);
    b->setMass(m_b);

    QED.addParticle(e);
    QED.addParticle(mu);
    QED.addParticle(t);
    QED.addParticle(u);
    QED.addParticle(d);
    QED.addParticle(s);
    QED.addParticle(tt);
    QED.addParticle(c);
    QED.addParticle(b);

    QED.renameParticle("A_em", "A");
    QED.refresh();

    // Straightforward, computes FeynmanRules for QED based on particle content and stuff.
    auto rules = ComputeFeynmanRules(QED);





    // particles_strings will have the list of particles involved in the process(es). It's provided when the program is called
    // externally from QED_loop_insertions.py (or other python script)
    // This block of code simply generates a vector of the appropriate mty::Insertions for the processes you want to calculate.
    std::vector<mty::Insertion> insertions;
    for (size_t i = 0; i!= particles_strings.size(); i++){
    insertions.push_back(get_insertion(particles_strings[i]));
    }


    mty::FeynOptions qed_opts = FeynOptions(); //mty::option::simplifyAmplitudes = false);
    qed_opts.simplifyAmplitudes = false;
    // qed_opts.abbreviateColorStructures = false;

    // ScopedProperty tempProperty(&csl::option::printIndexIds, false);
    auto process_ampl = QED.computeAmplitude(Order::OneLoop,  // OneLoop, TreeLevel
                                    insertions, 
                                    qed_opts // Provide FeynOptions to not abbreviate terms
    );
    std::vector<csl::Expr> ampl_expressions = {};               // Empty vector of scl::Expr for the amplitude expressions
    std::vector<csl::Expr> squared_ampl_expressions = square_amplitude_individually(process_ampl, QED);  //This computes |M|^2

    for (size_t i = 0; i!=process_ampl.size(); i++){                                        // For each element in process.ampl
    auto diagram_ampl_eval = Evaluated(process_ampl.expression(i), eval::abbreviation);     // Evaluate amplitude expression
    ampl_expressions.push_back(diagram_ampl_eval);                                          // Append evaluated amp to ampl_expressions
    }

    if (print_diagrams){
    Show(process_ampl);
    }



    //Exporting to files

    if (ampl_expressions.size() == 0){
    // don't create files if no amplitudes for process
    return 0;
    }

    // For now, don't calculate/output any squard amp stuff.
    // Or anything involving conversion to prefix notation - i.e., anything involving "raw"
    std::ofstream ampl_file_handle;
    std::ofstream sqampl_file_handle;
    // std::ofstream sqampl_raw_file_handle;
    // std::ofstream ampl_raw_file_handle;
    std::ofstream diagrams_file_handle;
    std::ofstream fdexpr_file_handle;
    std::ofstream loopint_file_handle;
    if (append_files){
    ampl_file_handle.open(amplitudes_file, std::ios_base::app);
    sqampl_file_handle.open(sqamplitudes_file, std::ios_base::app);
    // sqampl_raw_file_handle.open(sqamplitudes_raw_file, std::ios_base::app);
    // ampl_raw_file_handle.open(amplitudes_raw_file, std::ios_base::app);
    diagrams_file_handle.open(diagrams_file, std::ios_base::app);
    fdexpr_file_handle.open(fdexpr_file, std::ios_base::app);
    loopint_file_handle.open(loopint_file, std::ios_base::app);
    }
    else{
    ampl_file_handle.open(amplitudes_file);
    sqampl_file_handle.open(sqamplitudes_file);
    // sqampl_raw_file_handle.open(sqamplitudes_raw_file);
    // ampl_raw_file_handle.open(amplitudes_raw_file);
    diagrams_file_handle.open(diagrams_file);
    fdexpr_file_handle.open(fdexpr_file);
    loopint_file_handle.open(loopint_file);
    }

    // This converts amplitude expressions to prefix notation.
    // For the time being, comment this out. 
    // If to_prefix_notation were here, THAT would be the function which writes to a file/output stream.
    //  So I must figure out how to output the amplitudes as-is to ampl_file_haDisplayAbbndle
    // for(size_t i=0; i!=ampl_expressions.size(); i++){
    //     to_prefix_notation(ampl_expressions[i], ampl_file_handle);
    //     to_prefix_notation(squared_ampl_expressions[i], sqampl_file_handle);
       // sqampl_raw_file_handle << squared_ampl_expressions[i] << endl;
    //     ampl_raw_file_handle << ampl_expressions[i] << endl;
    // }

    for(size_t i=0; i!=ampl_expressions.size(); i++){
        ampl_file_handle << ampl_expressions[i] << endl;            // Output to open file. 
    }

    for(size_t i=0; i!=squared_ampl_expressions.size(); i++){
        sqampl_file_handle << squared_ampl_expressions[i] << endl;            // Output to open file. 
    }

    export_feynman_diagrams_str(process_ampl, fdexpr_file_handle);
    export_diagrams_str(process_ampl, diagrams_file_handle);
    get_loop_int_str(process_ampl, loopint_file_handle);
    // export_amflow_format(process_ampl, amflow_file_handle);

    ampl_file_handle.close();
    sqampl_file_handle.close();
    // sqampl_raw_file_handle.close();
    // ampl_raw_file_handle.close();
    diagrams_file_handle.close();
    fdexpr_file_handle.close();
    loopint_file_handle.close();

    DisplayAbbreviations();


	return 0;
}
