#include <resnet.h>
#include "resnet_backbone.h"

auto main() -> int
try
{
    using namespace dlib;
    // The ResNet50 backbone with custom activations and paddings
    resnet_backbone::train_50 net;
    // Load the weights for the the dlib ResNet50 architecture trained on ImageNet
    resnet::train_50 resnet50;
    deserialize("./resnet50_1000_imagenet_classifier.dnn") >> resnet50;
    layer<1>(net).layer_details() = layer<1>(resnet50).layer_details();
    layer<2>(net).layer_details() = layer<2>(resnet50).layer_details();
    // layer<3>(net).layer_details() = layer<3>(resnet50).layer_details(); // relu
    layer<4>(net).layer_details() = layer<4>(resnet50).layer_details();
    layer<5>(net).layer_details() = layer<5>(resnet50).layer_details();
    layer<6>(net).layer_details() = layer<6>(resnet50).layer_details();
    // layer<7>(net).layer_details() = layer<7>(resnet50).layer_details(); // relu
    layer<8>(net).layer_details() = layer<8>(resnet50).layer_details();
    layer<9>(net).layer_details() = layer<9>(resnet50).layer_details();
    // layer<10>(net).layer_details() = layer<10>(resnet50).layer_details(); // relu
     layer<11>(net).layer_details() = layer<11>(resnet50).layer_details();
    layer<12>(net).layer_details() = layer<12>(resnet50).layer_details();
    // layer<13>(net).layer_details() = layer<13>(resnet50).layer_details(); // tag
    // layer<14>(net).layer_details() = layer<14>(resnet50).layer_details(); // relu
    layer<15>(net).layer_details() = layer<15>(resnet50).layer_details();
    layer<16>(net).layer_details() = layer<16>(resnet50).layer_details();
    layer<17>(net).layer_details() = layer<17>(resnet50).layer_details();
    // layer<18>(net).layer_details() = layer<18>(resnet50).layer_details(); // relu
    layer<19>(net).layer_details() = layer<19>(resnet50).layer_details();
    layer<20>(net).layer_details() = layer<20>(resnet50).layer_details();
    // layer<21>(net).layer_details() = layer<21>(resnet50).layer_details(); // relu
    layer<22>(net).layer_details() = layer<22>(resnet50).layer_details();
    layer<23>(net).layer_details() = layer<23>(resnet50).layer_details();
    // layer<24>(net).layer_details() = layer<24>(resnet50).layer_details(); // tag
    // layer<25>(net).layer_details() = layer<25>(resnet50).layer_details(); // relu
    layer<26>(net).layer_details() = layer<26>(resnet50).layer_details();
    layer<27>(net).layer_details() = layer<27>(resnet50).layer_details();
    // layer<28>(net).layer_details() = layer<28>(resnet50).layer_details(); // skip
    // layer<29>(net).layer_details() = layer<29>(resnet50).layer_details(); // tag
    layer<30>(net).layer_details() = layer<30>(resnet50).layer_details();
    layer<31>(net).layer_details() = layer<31>(resnet50).layer_details();
    // layer<32>(net).layer_details() = layer<32>(resnet50).layer_details(); // relu
    layer<33>(net).layer_details() = layer<33>(resnet50).layer_details();
    layer<34>(net).layer_details().get_layer_params() = layer<34>(resnet50).layer_details().get_layer_params(); // con
    // layer<35>(net).layer_details() = layer<35>(resnet50).layer_details(); // relu
    layer<36>(net).layer_details() = layer<36>(resnet50).layer_details();
    layer<37>(net).layer_details() = layer<37>(resnet50).layer_details();
    // layer<38>(net).layer_details() = layer<38>(resnet50).layer_details(); // tag
    // layer<39>(net).layer_details() = layer<39>(resnet50).layer_details(); // relu
    layer<40>(net).layer_details() = layer<40>(resnet50).layer_details();
    layer<41>(net).layer_details() = layer<41>(resnet50).layer_details();
    layer<42>(net).layer_details() = layer<42>(resnet50).layer_details();
    // layer<43>(net).layer_details() = layer<43>(resnet50).layer_details(); // relu
    layer<44>(net).layer_details() = layer<44>(resnet50).layer_details();
    layer<45>(net).layer_details() = layer<45>(resnet50).layer_details();
    // layer<46>(net).layer_details() = layer<46>(resnet50).layer_details(); // relu
    layer<47>(net).layer_details() = layer<47>(resnet50).layer_details();
    layer<48>(net).layer_details() = layer<48>(resnet50).layer_details();
    // layer<49>(net).layer_details() = layer<49>(resnet50).layer_details(); // tag
    // layer<50>(net).layer_details() = layer<50>(resnet50).layer_details(); // relu
    layer<51>(net).layer_details() = layer<51>(resnet50).layer_details();
    layer<52>(net).layer_details() = layer<52>(resnet50).layer_details();
    layer<53>(net).layer_details() = layer<53>(resnet50).layer_details();
    // layer<54>(net).layer_details() = layer<54>(resnet50).layer_details(); // relu
    layer<55>(net).layer_details() = layer<55>(resnet50).layer_details();
    layer<56>(net).layer_details() = layer<56>(resnet50).layer_details();
    // layer<57>(net).layer_details() = layer<57>(resnet50).layer_details(); // relu
    layer<58>(net).layer_details() = layer<58>(resnet50).layer_details();
    layer<59>(net).layer_details() = layer<59>(resnet50).layer_details();
    // layer<60>(net).layer_details() = layer<60>(resnet50).layer_details(); // tag
    // layer<61>(net).layer_details() = layer<61>(resnet50).layer_details(); // relu
    layer<62>(net).layer_details() = layer<62>(resnet50).layer_details();
    layer<63>(net).layer_details() = layer<63>(resnet50).layer_details();
    layer<64>(net).layer_details() = layer<64>(resnet50).layer_details();
    // layer<65>(net).layer_details() = layer<65>(resnet50).layer_details(); // relu
    layer<66>(net).layer_details() = layer<66>(resnet50).layer_details();
    layer<67>(net).layer_details() = layer<67>(resnet50).layer_details();
    // layer<68>(net).layer_details() = layer<68>(resnet50).layer_details(); // relu
    layer<69>(net).layer_details() = layer<69>(resnet50).layer_details();
    layer<70>(net).layer_details() = layer<70>(resnet50).layer_details();
    // layer<71>(net).layer_details() = layer<71>(resnet50).layer_details(); // tag
    // layer<72>(net).layer_details() = layer<72>(resnet50).layer_details(); // relu
    layer<73>(net).layer_details() = layer<73>(resnet50).layer_details();
    layer<74>(net).layer_details() = layer<74>(resnet50).layer_details();
    layer<75>(net).layer_details() = layer<75>(resnet50).layer_details();
    // layer<76>(net).layer_details() = layer<76>(resnet50).layer_details(); // relu
    layer<77>(net).layer_details() = layer<77>(resnet50).layer_details();
    layer<78>(net).layer_details() = layer<78>(resnet50).layer_details();
    // layer<79>(net).layer_details() = layer<79>(resnet50).layer_details(); // relu
    layer<80>(net).layer_details() = layer<80>(resnet50).layer_details();
    layer<81>(net).layer_details() = layer<81>(resnet50).layer_details();
    // layer<82>(net).layer_details() = layer<82>(resnet50).layer_details(); // tag
    // layer<83>(net).layer_details() = layer<83>(resnet50).layer_details(); // relu
    layer<84>(net).layer_details() = layer<84>(resnet50).layer_details();
    layer<85>(net).layer_details() = layer<85>(resnet50).layer_details();
    layer<86>(net).layer_details() = layer<86>(resnet50).layer_details();
    // layer<87>(net).layer_details() = layer<87>(resnet50).layer_details(); // relu
    layer<88>(net).layer_details() = layer<88>(resnet50).layer_details();
    layer<89>(net).layer_details() = layer<89>(resnet50).layer_details();
    // layer<90>(net).layer_details() = layer<90>(resnet50).layer_details(); // relu
    layer<91>(net).layer_details() = layer<91>(resnet50).layer_details();
    layer<92>(net).layer_details() = layer<92>(resnet50).layer_details();
    // layer<93>(net).layer_details() = layer<93>(resnet50).layer_details(); // tag
    // layer<94>(net).layer_details() = layer<94>(resnet50).layer_details(); // relu
    layer<95>(net).layer_details() = layer<95>(resnet50).layer_details();
    layer<96>(net).layer_details() = layer<96>(resnet50).layer_details();
    // layer<97>(net).layer_details() = layer<97>(resnet50).layer_details(); // skip
    // layer<98>(net).layer_details() = layer<98>(resnet50).layer_details(); // tag
    layer<99>(net).layer_details() = layer<99>(resnet50).layer_details();
    layer<100>(net).layer_details() = layer<100>(resnet50).layer_details();
    // layer<101>(net).layer_details() = layer<101>(resnet50).layer_details(); // relu
    layer<102>(net).layer_details() = layer<102>(resnet50).layer_details();
    layer<103>(net).layer_details().get_layer_params() = layer<103>(resnet50).layer_details().get_layer_params(); // con
    // layer<104>(net).layer_details() = layer<104>(resnet50).layer_details(); // relu
    layer<105>(net).layer_details() = layer<105>(resnet50).layer_details();
    layer<106>(net).layer_details() = layer<106>(resnet50).layer_details();
    // layer<107>(net).layer_details() = layer<107>(resnet50).layer_details(); // tag
    // layer<108>(net).layer_details() = layer<108>(resnet50).layer_details(); // relu
    layer<109>(net).layer_details() = layer<109>(resnet50).layer_details();
    layer<110>(net).layer_details() = layer<110>(resnet50).layer_details();
    layer<111>(net).layer_details() = layer<111>(resnet50).layer_details();
    // layer<112>(net).layer_details() = layer<112>(resnet50).layer_details(); // relu
    layer<113>(net).layer_details() = layer<113>(resnet50).layer_details();
    layer<114>(net).layer_details() = layer<114>(resnet50).layer_details();
    // layer<115>(net).layer_details() = layer<115>(resnet50).layer_details(); // relu
    layer<116>(net).layer_details() = layer<116>(resnet50).layer_details();
    layer<117>(net).layer_details() = layer<117>(resnet50).layer_details();
    // layer<118>(net).layer_details() = layer<118>(resnet50).layer_details(); // tag
    // layer<119>(net).layer_details() = layer<119>(resnet50).layer_details(); // relu
    layer<120>(net).layer_details() = layer<120>(resnet50).layer_details();
    layer<121>(net).layer_details() = layer<121>(resnet50).layer_details();
    layer<122>(net).layer_details() = layer<122>(resnet50).layer_details();
    // layer<123>(net).layer_details() = layer<123>(resnet50).layer_details(); // relu
    layer<124>(net).layer_details() = layer<124>(resnet50).layer_details();
    layer<125>(net).layer_details() = layer<125>(resnet50).layer_details();
    // layer<126>(net).layer_details() = layer<126>(resnet50).layer_details(); // relu
    layer<127>(net).layer_details() = layer<127>(resnet50).layer_details();
    layer<128>(net).layer_details() = layer<128>(resnet50).layer_details();
    // layer<129>(net).layer_details() = layer<129>(resnet50).layer_details(); // tag
    // layer<130>(net).layer_details() = layer<130>(resnet50).layer_details(); // relu
    layer<131>(net).layer_details() = layer<131>(resnet50).layer_details();
    layer<132>(net).layer_details() = layer<132>(resnet50).layer_details();
    layer<133>(net).layer_details() = layer<133>(resnet50).layer_details();
    // layer<134>(net).layer_details() = layer<134>(resnet50).layer_details(); // relu
    layer<135>(net).layer_details() = layer<135>(resnet50).layer_details();
    layer<136>(net).layer_details() = layer<136>(resnet50).layer_details();
    // layer<137>(net).layer_details() = layer<137>(resnet50).layer_details(); // relu
    layer<138>(net).layer_details() = layer<138>(resnet50).layer_details();
    layer<139>(net).layer_details() = layer<139>(resnet50).layer_details();
    // layer<140>(net).layer_details() = layer<140>(resnet50).layer_details(); // tag
    // layer<141>(net).layer_details() = layer<141>(resnet50).layer_details(); // relu
    layer<142>(net).layer_details() = layer<142>(resnet50).layer_details();
    layer<143>(net).layer_details() = layer<143>(resnet50).layer_details();
    // layer<144>(net).layer_details() = layer<144>(resnet50).layer_details(); // skip
    // layer<145>(net).layer_details() = layer<145>(resnet50).layer_details(); // tag
    layer<146>(net).layer_details() = layer<146>(resnet50).layer_details();
    layer<147>(net).layer_details() = layer<147>(resnet50).layer_details();
    // layer<148>(net).layer_details() = layer<148>(resnet50).layer_details(); // relu
    layer<149>(net).layer_details() = layer<149>(resnet50).layer_details();
    layer<150>(net).layer_details().get_layer_params() = layer<150>(resnet50).layer_details().get_layer_params(); // con
    // layer<151>(net).layer_details() = layer<151>(resnet50).layer_details(); // relu
    layer<152>(net).layer_details() = layer<152>(resnet50).layer_details();
    layer<153>(net).layer_details() = layer<153>(resnet50).layer_details();
    // layer<154>(net).layer_details() = layer<154>(resnet50).layer_details(); // tag
    // layer<155>(net).layer_details() = layer<155>(resnet50).layer_details(); // relu
    layer<156>(net).layer_details() = layer<156>(resnet50).layer_details();
    layer<157>(net).layer_details() = layer<157>(resnet50).layer_details();
    layer<158>(net).layer_details() = layer<158>(resnet50).layer_details();
    // layer<159>(net).layer_details() = layer<159>(resnet50).layer_details(); // relu
    layer<160>(net).layer_details() = layer<160>(resnet50).layer_details();
    layer<161>(net).layer_details() = layer<161>(resnet50).layer_details();
    // layer<162>(net).layer_details() = layer<162>(resnet50).layer_details(); // relu
    layer<163>(net).layer_details() = layer<163>(resnet50).layer_details();
    layer<164>(net).layer_details() = layer<164>(resnet50).layer_details();
    // layer<165>(net).layer_details() = layer<165>(resnet50).layer_details(); // tag
    // layer<166>(net).layer_details() = layer<166>(resnet50).layer_details(); // relu
    layer<167>(net).layer_details() = layer<167>(resnet50).layer_details();
    layer<168>(net).layer_details() = layer<168>(resnet50).layer_details();
    layer<169>(net).layer_details() = layer<169>(resnet50).layer_details();
    // layer<170>(net).layer_details() = layer<170>(resnet50).layer_details(); // relu
    layer<171>(net).layer_details() = layer<171>(resnet50).layer_details();
    layer<172>(net).layer_details() = layer<172>(resnet50).layer_details();
    // layer<173>(net).layer_details() = layer<173>(resnet50).layer_details(); // relu
    layer<174>(net).layer_details() = layer<174>(resnet50).layer_details();
    layer<175>(net).layer_details() = layer<175>(resnet50).layer_details();
    // layer<176>(net).layer_details() = layer<176>(resnet50).layer_details(); // tag
    // layer<177>(net).layer_details() = layer<177>(resnet50).layer_details(); // relu
    layer<178>(net).layer_details() = layer<178>(resnet50).layer_details();
    layer<179>(net).layer_details() = layer<179>(resnet50).layer_details();
    layer<180>(net).layer_details() = layer<180>(resnet50).layer_details();
    // layer<181>(net).layer_details() = layer<181>(resnet50).layer_details(); // relu
    layer<182>(net).layer_details() = layer<182>(resnet50).layer_details();
    layer<183>(net).layer_details() = layer<183>(resnet50).layer_details();
    // layer<184>(net).layer_details() = layer<184>(resnet50).layer_details(); // relu
    layer<185>(net).layer_details() = layer<185>(resnet50).layer_details();
    layer<186>(net).layer_details() = layer<186>(resnet50).layer_details();
    // layer<187>(net).layer_details() = layer<187>(resnet50).layer_details(); // tag
    // layer<188>(net).layer_details() = layer<188>(resnet50).layer_details(); // max_pool
    // layer<189>(net).layer_details() = layer<189>(resnet50).layer_details(); // relu
    layer<190>(net).layer_details() = layer<190>(resnet50).layer_details();
    layer<191>(net).layer_details().get_layer_params() = layer<191>(resnet50).layer_details().get_layer_params(); // con
    // layer<192>(net).layer_details() = layer<192>(resnet50).layer_details(); // input

    matrix<rgb_pixel> dummy(608, 608);
    net(dummy);
    std::cout << net << '\n';
    net.clean();
    std::cout << "computational layers: " << net.subnet().subnet().subnet().num_computational_layers << '\n';
    std::cout << "number of parameters: " << count_parameters(net.subnet().subnet().subnet()) << '\n';
    // serialize the backbone only: remove loss, fc and avg_pool_everythng
    serialize("resnet50_pretrained_backbone.dnn") << net.subnet().subnet().subnet();

    return EXIT_SUCCESS;

}
catch(const std::exception& e)
{
    std::cout << e.what() << '\n';
    return EXIT_FAILURE;
}

