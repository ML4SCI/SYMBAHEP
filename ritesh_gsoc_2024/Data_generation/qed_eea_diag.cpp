#include "marty.h"
#include <fstream>
#include <string_view>
#include <string>
#include <vector>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <condition_variable>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
using namespace std;
using namespace csl;
using namespace mty;
using namespace std::chrono;

std::ofstream dtout;


bool isInternalVertex(csl::Tensor const &X)
{
    const auto &name = X->getName();            //dereferences member getName() of input tensor reference X, sets it to reference name
    return !name.empty() && name[0] == 'V';     //If name not empty and the first character is 'V', this is an internal vertex.
}

struct SimpleField
{
    std::string name;
    std::string vertex;
    bool p;
    bool s;
};

void convertField(mty::QuantumField const &field, SimpleField &result)
{
    result.name = field.getName();
    result.vertex = field.getPoint()->getName();
    result.p = field.isParticle();
    result.s = field.isOnShell();
}

// Connection objects represent and list fields that connect
// to the same vertex
struct Connection {
    std::string vertex;
    std::vector<SimpleField> externalFields; // List of fields connected to this vertex - not necessarily the external fields (insertions)!
};

std::vector<Connection> getDiagramConnections(std::vector<std::shared_ptr<mty::wick::Node>> const &nodes)
{
    std::unordered_map<std::string, Connection> connections;        // connections - keys are strings, values are connection structs.
    for (const auto &node : nodes) {                                // iterate through references to nodes in input nodes
        auto field = *node->field;                                  // get field represented by node in the graph
        auto partner = *node->partner.lock()->field;                // get contracted partner field, save as "partner"
        bool fieldInternal = isInternalVertex(field.getPoint());    // Check if current/partner field is on internal vertex
        bool partnerInternal = isInternalVertex(partner.getPoint());// what does getPoint actually do though???
        
        SimpleField convertedField;
        
        if (partnerInternal)                                        // if partner is internal
        {
            csl::Tensor vertex;                                     // create csl::Tensor object
            if (fieldInternal)                                      // If field also internal
                vertex = field.getPoint();                          // get corresponding internal vertex for field
            else
                vertex = partner.getPoint();                        // get corresponding internal vertex for contracted partner
            auto pos = connections.find(vertex->getName());         // look for specific vertex in connections using getName()
            convertField(field, convertedField);                    // Convert field to SimpleField
            if (pos == connections.end()) {
                // if not in connections, Add new Connection {vertex name, converted field} to end of connections with the key vertex->getName()
                connections[vertex->getName()] = Connection{vertex->getName(), {convertedField}};
            }
            else {
                // if in connections, Add converted field to externalFields vector attr of connections[vertex->getName()]
                connections[vertex->getName()].externalFields.push_back(convertedField);
            }
        }
        
        if (fieldInternal)                                          // if field itself is internal
        {
            csl::Tensor vertex;                                     // create csl::Tensor object
            if (partnerInternal)                                    // If partner also internal
                vertex = partner.getPoint();                        // get corresponding internal vertex for partner
            else
                vertex = field.getPoint();                          // get corresponding internal vertex for field
            auto pos = connections.find(vertex->getName());         // look for specific vertex in connections using getName()
            convertField(partner, convertedField);                  // Convert partner to SimpleField
            if (pos == connections.end()) {
                // if not in connections, Add new Connection{vertex name, converted partner} to end of connections with the key vertex->getName()
                connections[vertex->getName()] = Connection{vertex->getName(), {convertedField}};
            }
            else {
                // if in connections, Add converted partner to externalFields vector attr of connections[vertex->getName()]
                connections[vertex->getName()].externalFields.push_back(convertedField);
            }
        }
    }
    
    std::vector<Connection> res;                    // Vector of connection objects
    res.reserve(connections.size());                // Make it the same size as connections
    for (const auto &el : connections) {            // for each element in connections, i.e. each
        res.push_back(std::move(el.second));        // move 2nd element of el (i.e., the connection itself) from to res
    }

    return res;
}

std::vector<std::vector<Connection>> processAmplitudes(Amplitude const &ampl)
// Takes in a reference to an Amplitude object (mty::Amplitude?) &ampl
// Returns a vector of vectors of connection objects.
{
    // Get diagrams
    std::vector<FeynmanDiagram> const &diagrams = ampl.getDiagrams();   // getDiagrams(from the input amplitude)
    std::vector<std::vector<Connection>> topologies;                    // setup a vector of connection vectors - topologies
    topologies.reserve(diagrams.size());                                // topologies should be same size as # of diagrams
    for (auto const &diag : diagrams) {                                 // for each diagram
        //std::cout << "\n************\n";
        //std::cout << "New diagram:\n";
        // Getes the nodes of the diagram
        // (contains edges by looking at the nodes' partners)
        auto nodes = diag.getDiagram()->getNodes();                     // Get nodes from the diagram
        auto connections = getDiagramConnections(nodes);                // get connections from diagram nodes
        topologies.push_back(std::move(connections));                   // add vector of connections to topologies
    }
    return topologies;                                                  // return full vector topologies
}

void export_diagrams_str(mty::Amplitude &ampl, std::ofstream &stream)                    
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
                // stream << (field.p ? "Particle" : "AntiParticle") << (field.s ? " OnShell " : " OffShell ") <<field.name << "(" << field.vertex << "),";
                stream << (field.p ? "" : "AntiPart ") << (field.s ? "" : " OffShell ") <<field.name << "(" << field.vertex << "), ";
            }
            // stream << endl;
        }
        // stream << "--------------" << endl;
    }
}

string getParticleString(const Insertion& insertion) {
    string particle_string;
    
    if (insertion.isOnShell()) {
        if (insertion.isParticle()) {
            particle_string = "";
        } else {
            particle_string = "AntiPart ";
        }
    } else {
        if (insertion.isParticle()) {
            particle_string = "OffShell ";
        } else {
            particle_string = "AntiPart OffShell ";
        }
    }
    
    particle_string += toString(GetExpression(insertion));
    return particle_string;
}

template<typename Func>
auto callWithTimeout(std::chrono::minutes timeout_duration, Func func) -> decltype(func()) {
    std::mutex m;
    std::condition_variable cv;
    bool completed = false;
    std::optional<decltype(func())> result;

    std::function<decltype(func())()> task = func;
    std::thread t([&cv, &result, &completed, task]() {
        result.emplace(task());
        completed = true;
        cv.notify_one();
    });

    std::unique_lock<std::mutex> l(m);
    if (!cv.wait_for(l, timeout_duration, [&completed]{ return completed; })) {
        t.detach(); // Detach if still running and timeout occurred
        throw std::runtime_error("Timeout");
    } else {
        t.join(); // Ensure the thread finishes before returning
    }

    return *result;
}

size_t lengthOfPrintedText(const csl::Expr& expr) {
    std::ostringstream oss;
    oss << expr;
    return oss.str().length();
}

void evaluateAndPrint(Model& model, std::vector<std::string>& parts, const std::vector<Insertion>& insertions, const int& numIn, Order& order) {
    try {
        auto ampl = model.computeAmplitude(order, insertions);
        
        if (!ampl.empty()) {
            for (size_t n = 0; n != ampl.size(); n++) {
                // Expr term = ampl.expression(n);
                // Evaluate(term, csl::eval::abbreviation);

                dtout << "Interaction: ";
                for (int i = 0; i < numIn; i++) {
                    dtout << " " << getParticleString(insertions[i]) << " ";
                }
                
                dtout << " to ";
                
                for (size_t j = numIn; j < parts.size(); j++) {
                    dtout << " " << getParticleString(insertions[j]) << " ";
                }

                Amplitude newAmpl = Amplitude(ampl.getOptions(), {ampl.getDiagrams()[n]}, ampl.getKinematics());
                Expr term = newAmpl.expression(0);
                
                const std::size_t maxLength = 2000;

                if (lengthOfPrintedText(term) > maxLength) {
                    throw std::runtime_error("Error: The string is too long.");
                    dtout << ":  :  : ";
                }

                Evaluate(term, csl::eval::abbreviation);

                dtout << ": ";

                export_diagrams_str(newAmpl, dtout);

                dtout << ": " << term << " : ";

                Expr squaredAmpl = model.computeSquaredAmplitude(newAmpl);
                
                Evaluate(squaredAmpl, csl::eval::abbreviation);
                
                dtout << squaredAmpl << '\n';
            }
        }
    } catch (CSLError& e) {
        std::cout << "Caught CSLError exception: " << e << std::endl;
    } catch (const std::exception& e) {
        // Log the error and continue with the next combination
        dtout << "Error evaluating combination: ";
        for (int i = 0; i < numIn; i++) {
            dtout << " " << getParticleString(insertions[i]) << " ";
        }
        
        dtout << " to ";
        
        for (size_t j = numIn; j < parts.size(); j++) {
            dtout << " " << getParticleString(insertions[j]) << " ";
        }
        dtout << ": " << e.what() << '\n';
    } catch (...) {
        std::cout << "Caught unknown exception" << std::endl;
    }
}

void populateOptions(std::vector<Insertion>& options, const std::string& part, const bool& isIncoming) {
    options.clear(); // Clear the vector to avoid retaining previous elements
    if (isIncoming) {
        options.push_back(Incoming(part));
        options.push_back(Incoming(OffShell(part)));
        options.push_back(Incoming(AntiPart(part)));
        options.push_back(Incoming(AntiPart(OffShell(part))));
    } else {
        options.push_back(Outgoing(part));
        options.push_back(Outgoing(OffShell(part)));
        options.push_back(Outgoing(AntiPart(part)));
        options.push_back(Outgoing(AntiPart(OffShell(part))));
    }
}

void generateCombinations(Model& model, std::vector<std::string>& parts, std::vector<Insertion>& insertions, std::vector<Insertion>& options, const int& index, const int& numIn, const int& numOut, Order& order) {
    if (index == numIn + numOut) {
        evaluateAndPrint(model, parts, insertions, numIn, order);
        return;
    }

    populateOptions(options, parts[index], index < numIn);

    for (size_t i = 0; i < options.size(); i++) {
        std::vector<Insertion> tempOptions = std::move(options);
        Insertion previousInsertion = std::move(insertions[index]);
        insertions[index] = tempOptions[i];
        generateCombinations(model, parts, insertions, options, index + 1, numIn, numOut, order);
        options = std::move(tempOptions);
        insertions[index] = std::move(previousInsertion);
    }
}

int recursive_loop(const int& max_loops, const int& current_loop, const int& range, Model& model, const int& numIn, const int& numOut, Order& order, const int& firstLoopIdx, std::vector<std::string>& parts, std::vector<Insertion>& insertions, std::vector<Insertion>& options, const std::vector<std::string_view>& lis, const int& start_index = 0) {
    if (current_loop == max_loops) {
        try {
            generateCombinations(model, parts, insertions, options, 0, numIn, numOut, order);
        } catch (const std::exception& e) {
            std::cerr << "Error in main loop: " << e.what() << '\n';
        }
        return current_loop;
    } else {
        if (current_loop == 0) {
            std::vector<std::string> original_parts(parts.begin() + current_loop, parts.end());
            for (int i = firstLoopIdx; i < firstLoopIdx + 1; i++) {
                std::string original_part = std::move(parts[current_loop]);
                parts[current_loop] = std::string(lis[i]);
                recursive_loop(max_loops, current_loop + 1, range, model, numIn, numOut, order, firstLoopIdx, parts, insertions, options, lis, i);
                parts[current_loop] = std::move(original_part);
            }
        } else {
            int loop_range = (current_loop == 0 || current_loop == numIn) ? 0 : start_index;
            std::vector<std::string> original_parts(parts.begin() + current_loop, parts.end());
            for (int i = loop_range; i < range; i++) {
                std::string original_part = std::move(parts[current_loop]);
                parts[current_loop] = std::string(lis[i]);
                recursive_loop(max_loops, current_loop + 1, range, model, numIn, numOut, order, firstLoopIdx, parts, insertions, options, lis, i);
                parts[current_loop] = std::move(original_part);
            }
        }
    }
    return current_loop;
}

int com_amp(Model& model, std::string type, const std::vector<std::string_view>& lis, const int& numParts, const int& numIn, const int& numOut, Order& order, std::string& orderStr, const int& firstLoopIdx) {
    // Format the filename to include the number of incoming and outgoing particles
    std::stringstream filename;
    filename << "/output/" << type << "-" << numIn << "-to-" << numOut << "-diag-" << orderStr << "-" << firstLoopIdx << ".txt";
    dtout.open(filename.str());

    std::vector<Insertion> options (4, Incoming(AntiPart(OffShell(lis[1]))));
    std::vector<Insertion> insertions(numIn + numOut, Incoming(AntiPart(OffShell(lis[1]))));
    std::vector<std::string> parts(numIn + numOut, "xyz");
    
    recursive_loop(numIn + numOut, 0, numParts, model, numIn, numOut, order, firstLoopIdx, parts, insertions, options, lis);

    dtout.close();
    return 0;
}

Order parseOrder(const std::string& orderStr) {
    if (orderStr == "TreeLevel") {
        return Order::TreeLevel;
    } else if (orderStr == "OneLoop") {
        return Order::OneLoop;
    } else {
        throw std::invalid_argument("Invalid order: " + orderStr);
    }
}

int main(int argc, char *argv[]) {
    Model QED_Model;

    Expr psi = constant_s("e");
    QED_Model.addGaugedGroup(group::Type::U1, "em", psi);

    QED_Model.init();

    Particle e = diracfermion_s("e", QED_Model);
    Particle mu = diracfermion_s("mu", QED_Model);
    Particle t = diracfermion_s("t", QED_Model);
    Particle u = diracfermion_s("u", QED_Model);
    Particle d = diracfermion_s("d", QED_Model);
    Particle s = diracfermion_s("s", QED_Model);
    Particle tt = diracfermion_s("tt", QED_Model);
    Particle c = diracfermion_s("c", QED_Model);
    Particle b = diracfermion_s("b", QED_Model);



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


    QED_Model.addParticle(e);
    QED_Model.addParticle(mu);
    QED_Model.addParticle(t);
    QED_Model.addParticle(u);
    QED_Model.addParticle(d);
    QED_Model.addParticle(s);
    QED_Model.addParticle(tt);
    QED_Model.addParticle(c);
    QED_Model.addParticle(b);



    QED_Model.renameParticle("A_em", "A");

    QED_Model.refresh();

    std::vector<std::string_view> lis = {"e", "A"};
    
    int num_in = std::stoi(argv[1]);
    int num_out = std::stoi(argv[2]);
    std::string orderStr = argv[3];
    Order order = parseOrder(argv[3]);
    int loop_index = std::stoi(argv[4]);
    
    com_amp(QED_Model, "EEA", lis, 2, num_in, num_out, order, orderStr, loop_index);

}
