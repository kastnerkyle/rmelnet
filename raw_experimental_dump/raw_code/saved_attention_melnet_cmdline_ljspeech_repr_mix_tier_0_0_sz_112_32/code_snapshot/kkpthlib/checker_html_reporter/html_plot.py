# Author: Kyle Kastner
# webpage parts based heavily on examples from indirajhenny
# http://bl.ocks.org/indirajhenny/15d7218cadf4fa96e407f87f034b1ff7
import os

pre = '''
var notes = [
'''

pre_chunk = '''{
'''

chunk = '''    "name": "{}",
    "midi": {},
    "time": {},
    "velocity": {},
    "duration": {},
    "color": "{}",
    "highlight": "{}"
'''

post_chunk = '''},
'''

post = '''
];
'''

# need to auto-generate this...
code_stub_pre = '''
var midiOnly = [];

notes.forEach(function(item) {
    //get value of name
    var val = item.midi;
    //push duration value into array
    midiOnly.push(val);
});

//console.log(midiOnly);
'''

# create empty lists, as many as we have note range (start with 21?)
# create notes function and final function

N_LANES = 108 - 20

midi_to_name_lookup = {}
note_bases = {0: "C",
              1: "C#",
              2: "D",
              3: "Eb",
              4: "E",
              5: "F",
              6: "F#",
              7: "G",
              8: "Ab",
              9: "A",
              10: "Bb",
              11: "B"}

for i in range(20, 108):
    b = note_bases[i % 12]
    o = (i // 12) - 1
    midi_to_name_lookup[i] = b + str(o)

"""
LANES_LOOKUP= {20: "C3",
               19: "D3",
               18: "E3",
               17: "F3",
               16: "G3",
               15: "A3",
               14: "B3",

               13: "C4",
               12: "D4",
               11: "E4",
               10: "F4",
               9: "G4",
               8: "A4",
               7: "B4",

               6: "C5",
               5: "D5",
               4: "E5",
               3: "F5",
               2: "G5",
               1: "A5",
               0: "B5",
              }
"""
LANES_LOOKUP = {}
for _n, i in enumerate(range(20, 108)):
    rev = list(range(20, 108))[::-1]
    LANES_LOOKUP[i - 20] = midi_to_name_lookup[rev[_n]]

def _create_code_stub_init():
    s = "\n"
    template = "minimalNotes{} = [];\n"
    for i in range(N_LANES):
        s = s + template.format(i)

    template = "minimalNotesHighlight{} = [];\n"
    for i in range(N_LANES):
        s = s + template.format(i)
    s = s + "minimalNotesFinal = [];\n"
    s = s + "minimalNotesHighlightFinal = [];\n"
    return s
code_stub_init = _create_code_stub_init()

def _create_code_stub_functions():
    # use simple subs because the string has both {} and ()
    from string import Template
    s = "\n"
    
    template = '''
minimalNotes$index = notes.filter(function (el){
    return ((el.name == "$note_name") && (el.highlight == "no"));
});

var minimalNotesFinal$index = minimalNotes$index.map(function(el){
    var addKey = Object.assign({}, el);
    addKey.lane = $lane_index;
    return addKey;
});

minimalNotesHighlight$index = notes.filter(function (el){
    return ((el.name == "$note_name") && (el.highlight == "yes"));
});

var minimalNotesHightlightFinal$index = minimalNotesHighlight$index.map(function(el){
    var addKey = Object.assign({}, el);
    addKey.lane = $lane_index;
    return addKey;
});
'''
    t = Template(template)
    for i in range(N_LANES):
        s = s + t.substitute(index=i, lane_index=i, note_name=LANES_LOOKUP[i])
    return s

code_stub_functions = _create_code_stub_functions()

def _create_code_stub_combine():
    s = "\n"
    template = "minimalNotesFinal = minimalNotesFinal.concat(minimalNotesFinal{})\n"
    template += "minimalNotesHighlightFinal = minimalNotesHighlightFinal.concat(minimalNotesHightlightFinal{})\n"
    for i in range(N_LANES):
        s = s + template.format(i, i)
    s = s + "console.log(minimalNotesFinal)\n"
    s = s + "console.log(minimalNotesHighlightFinal)\n"
    return s

code_stub_combine = _create_code_stub_combine()

def make_chunk(note_tuple, highlight=False, highlight_color="red"):
    midi = note_tuple[0]
    name = midi_to_name_lookup[midi]
    start_time = note_tuple[1]
    velocity = 1.0
    duration = note_tuple[2]
    if highlight == True:
        highlight = "yes"
        color = highlight_color
    else:
        highlight = "no"
        color = "black"
    s = pre_chunk + chunk.format(name, midi, start_time, velocity, duration, color, highlight) + post_chunk
    return s

code_stub = code_stub_pre + code_stub_init + code_stub_functions + code_stub_combine
def make_plot_json(list_of_notes, notes_to_highlight=None, match_note_color="red"):
    # notes to highlight should be the same length as list_of_notes, each entry True, False, or the name of a specific color
    cur = pre
    # voice track?
    for _n, note in enumerate(list_of_notes):
        if notes_to_highlight is not None:
            if notes_to_highlight[_n] == False:
                c = make_chunk(note, highlight=notes_to_highlight[_n], highlight_color=match_note_color)
            elif notes_to_highlight[_n] == True:
                c = make_chunk(note, highlight=notes_to_highlight[_n], highlight_color=match_note_color)
            else:
                c = make_chunk(note, highlight=True, highlight_color=notes_to_highlight[_n])
        else:
            c = make_chunk(note, highlight=False, highlight_color=match_note_color)
        cur = cur + c
    return cur[:-2] + post + code_stub


def make_website_string(javascript_note_data_string, page_name="Piano Roll Plot", end_time=120, info_tag=None, report_index_value=0, base64_midi=None, midi_name=None):
    from string import Template
    plot_module_path = __file__
    plot_module_dir = str(os.sep).join(os.path.split(plot_module_path)[:-1])
    with open(plot_module_dir + os.sep + "report_template.html", "r") as f:
        l = f.read()
    t = Template(l)
    button_function = """
    function toggleReport%sFunction() {
        var x = document.getElementById("report%sinfo");
        if (x.style.display === "none") {
            x.style.display = "block";
        } else {
            x.style.display = "none";
        }
    }
    """ % (str(report_index_value), str(report_index_value))
    button_html = '<button onclick="toggleReport%sFunction()">Toggle Report %s Info</button>' % (str(report_index_value), str(report_index_value))
    button_html = ""
    info_tag_core = info_tag

    info_tag = '<div class="section" id="report%sinfo" display="block">\n' % str(report_index_value)
    if info_tag_core is None:
        info_tag += ""
    else:
        info_tag += info_tag_core
    info_tag += '\n</div>\n'
    # if we reverse the list, we reverse the axis
    report_string = t.substitute(PAGE_NAME=page_name, JAVASCRIPT_NOTE_DATA=javascript_note_data_string, LANE_NAMES=str([LANES_LOOKUP[i] for i in range(N_LANES)]), LANE_TIME_END=end_time, INFO_TAG=info_tag, BUTTON_HTML=button_html, BUTTON_FUNCTION=button_function, REPORT_NAME="report{}".format(report_index_value))

    with open(plot_module_dir + os.sep + "midi_player_template.html", "r") as f:
        l = f.read()
    t2 = Template(l)

    option_file_str = ''
    if midi_name is not None:
        option_file_str += '<option value="{}">{}</option>\n'.format(base64_midi, midi_name)
        midi_player_string = t2.substitute(MIDI_FILES_OPTION_LIST=option_file_str)
    else:
        midi_player_string = ''

    midijs_str = '<script type="text/javascript">\n'
    with open(plot_module_dir + os.sep + "MIDIFile.js", "r") as f:
        midijs_str += f.read()
    midijs_str += '</script>\n'

    webaudio_str = '<script type="text/javascript">\n'
    with open(plot_module_dir + os.sep + "WebAudioFontPlayer.js", "r") as f:
        webaudio_str += f.read()
    webaudio_str += '</script>\n'
    return midijs_str + webaudio_str + midi_player_string + report_string


def make_index_html_string2(base64_midis, midi_names, all_javascript_note_info, all_match_info):
    from string import Template
    plot_module_path = __file__
    plot_module_dir = str(os.sep).join(os.path.split(plot_module_path)[:-1])
    with open(plot_module_dir + os.sep + "midi_player_template.html", "r") as f:
        l = f.read()
    t2 = Template(l)

    option_file_str = ''
    for _i in range(len(midi_names)):
        midi_name = midi_names[_i]
        base64_midi = base64_midis[_i]
        option_file_str += '<option value="{}">{}</option>\n'.format(base64_midi, midi_name)
    midi_player_string = t2.substitute(MIDI_FILES_OPTION_LIST=option_file_str)

    midijs_str = '<script type="text/javascript">\n'
    with open(plot_module_dir + os.sep + "MIDIFile.js", "r") as f:
        midijs_str += f.read()
    midijs_str += '</script>\n'

    webaudio_str = '<script type="text/javascript">\n'
    with open(plot_module_dir + os.sep + "WebAudioFontPlayer.js", "r") as f:
        webaudio_str += f.read()
    webaudio_str += '</script>\n'
    # need to do midi player first, with ALL audio file names...
    # then div + buttons

    with open(plot_module_dir + os.sep + "report_template.html", "r") as f:
        l = f.read()
    t = Template(l)
    final_report_string = ""
    selection_dropdown_string_core = ""
    for _i in range(len(all_javascript_note_info)):
        report_index_value = _i
        page_name="Piano Roll Plot"
        end_time=240
        javascript_note_data_string=all_javascript_note_info[_i]

        button_function = """
        function toggleReport%sFunction() {
            var x = document.getElementById("biggrid$REPORT_NAME");
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }

            var x = document.getElementById("bigchart$REPORT_NAME");
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }

            var x = document.getElementById("biginfo$REPORT_NAME");
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }

            var x = document.getElementById("bigvert$REPORT_NAME");
            if (x.style.display === "none") {
                x.style.display = "block";
            } else {
                x.style.display = "none";
            }

            var els = document.getElementsByClassName('vertical');
            for (var i = 0; i < els.length; i++) {
                x = els.item(i)
                if (x.style.display === "none") {
                    x.style.display = "block";
                }
            }
        }

        function clearReport%sFunction() {
            var x = document.getElementById("biggrid$REPORT_NAME");
            x.style.display = "none";

            var x = document.getElementById("bigchart$REPORT_NAME");
            x.style.display = "none";

            var x = document.getElementById("biginfo$REPORT_NAME");
            x.style.display = "none";

            var x = document.getElementById("bigvert$REPORT_NAME");
            x.style.display = "none";

            var els = document.getElementsByClassName('vertical');
            for (var i = 0; i < els.length; i++) {
                x = els.item(i)
                if (x.style.display === "block") {
                    x.style.display = "none";
                }
            }
        }

        function reportChangeFunc1() {
            var reportBoxShadow = document.getElementById('reportBoxShadow');
            var reportBox1 = document.getElementById('reportBox1');
            var reportBox2 = document.getElementById('reportBox2');
            var reportBox3 = document.getElementById('reportBox3');
            for(i = 1; i < reportBoxShadow.length; i++) {
                // resets the values to defaults before setting new changes
                // skip 0 elem
                if ((i != reportBox2.selectedIndex) && (i != reportBox3.selectedIndex)) {
                    try {
                        eval("clearReport" + (i - 1) + "Function();");
                    } catch(err) {

                    };
                }
                reportBox1.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
                reportBox2.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
                reportBox3.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
            }

            reportBox2.options[reportBox1.selectedIndex].innerHTML = "--";
            reportBox3.options[reportBox1.selectedIndex].innerHTML = "---";

            reportBox1.options[reportBox2.selectedIndex].innerHTML = "-";
            reportBox3.options[reportBox2.selectedIndex].innerHTML = "---";

            reportBox1.options[reportBox3.selectedIndex].innerHTML = "-";
            reportBox2.options[reportBox3.selectedIndex].innerHTML = "--";

            eval(reportBox1.options[reportBox1.selectedIndex].value + "();");
        }

        function reportChangeFunc2() {
            var reportBoxShadow = document.getElementById('reportBoxShadow');
            var reportBox1 = document.getElementById('reportBox1');
            var reportBox2 = document.getElementById('reportBox2');
            var reportBox3 = document.getElementById('reportBox3');
            for(i = 1; i < reportBoxShadow.length; i++) {
                // resets the values to defaults before setting new changes
                // skip 0 element
                if ((i != reportBox1.selectedIndex) && (i != reportBox3.selectedIndex)) {
                    try {
                        eval("clearReport" + (i - 1) + "Function();");
                    } catch(err) {

                    };
                }
                reportBox1.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
                reportBox2.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
                reportBox3.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
            }

            reportBox2.options[reportBox1.selectedIndex].innerHTML = "--";
            reportBox3.options[reportBox1.selectedIndex].innerHTML = "---";

            reportBox1.options[reportBox2.selectedIndex].innerHTML = "-";
            reportBox3.options[reportBox2.selectedIndex].innerHTML = "---";

            reportBox1.options[reportBox3.selectedIndex].innerHTML = "-";
            reportBox2.options[reportBox3.selectedIndex].innerHTML = "--";

            eval(reportBox2.options[reportBox2.selectedIndex].value + "();");
        }

        function reportChangeFunc3() {
            var reportBoxShadow = document.getElementById('reportBoxShadow');
            var reportBox1 = document.getElementById('reportBox1');
            var reportBox2 = document.getElementById('reportBox2');
            var reportBox3 = document.getElementById('reportBox3');

            for(i = 1; i < reportBoxShadow.length; i++) {
                // resets the values to defaults before setting new changes
                // skip 0 "select" elem
                if ((i != reportBox1.selectedIndex) && (i != reportBox2.selectedIndex)) {
                    try {
                        eval("clearReport" + (i - 1) + "Function();");
                    } catch(err) {

                    };
                }
                reportBox1.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
                reportBox2.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
                reportBox3.options[i].innerHTML = reportBoxShadow.options[i].innerHTML
            }

            reportBox2.options[reportBox1.selectedIndex].innerHTML = "--";
            reportBox3.options[reportBox1.selectedIndex].innerHTML = "---";

            reportBox1.options[reportBox2.selectedIndex].innerHTML = "-";
            reportBox3.options[reportBox2.selectedIndex].innerHTML = "---";

            reportBox1.options[reportBox3.selectedIndex].innerHTML = "-";
            reportBox2.options[reportBox3.selectedIndex].innerHTML = "--";

            eval(reportBox3.options[reportBox3.selectedIndex].value + "();");
        };

        function clearReportFunction() {
            var reportBox1 = document.getElementById('reportBox1');
            var reportBox2 = document.getElementById('reportBox2');
            var reportBox3 = document.getElementById('reportBox3');
            try {
                eval(reportBox1.options[reportBox1.selectedIndex].value + "();");
                eval(reportBox2.options[reportBox2.selectedIndex].value + "();");
                eval(reportBox3.options[reportBox3.selectedIndex].value + "();");
            } catch (err) {

            };

            var reportBoxShadow = document.getElementById('reportBoxShadow');
            for(i = 1; i < reportBoxShadow.length; i++) {
                // resets the values to defaults before setting new changes
                // skip 0 "select" elem
                reportBox1.options[i].innerHTML = reportBoxShadow.options[i].innerHTML;
                reportBox2.options[i].innerHTML = reportBoxShadow.options[i].innerHTML;
                reportBox3.options[i].innerHTML = reportBoxShadow.options[i].innerHTML;
            };

            reportBox1.selectedIndex = 0;
            reportBox2.selectedIndex = 0;
            reportBox3.selectedIndex = 0;
        };

        var x = document.getElementById("reportBoxShadow");
        x.style.display = "none";

        var x = document.getElementById("biggrid$REPORT_NAME");
        x.style.display = "none";
        var x = document.getElementById("bigchart$REPORT_NAME");
        x.style.display = "none";
        var x = document.getElementById("bigvert$REPORT_NAME");
        x.style.display = "none";
        var x = document.getElementById("biginfo$REPORT_NAME");
        x.style.display = "none";
        """ % (str(report_index_value), str(report_index_value))

        bt = Template(button_function)
        button_function = bt.substitute(REPORT_NAME="report{}".format(report_index_value))

        match_info = all_match_info[_i]

        selection_dropdown_string_core += '        <option value="toggleReport%sFunction">%s</option>\n' % (str(report_index_value), match_info[0].split(os.sep)[-1])

        button_html = '<button position="relative" onclick="toggleReport%sFunction()">Toggle Report %s Info</button>' % (str(report_index_value), str(report_index_value))

        '''
        if report_index_value != 0:
            # this is NOT constant... need to dig on the data
            # for now, don't show the bar for anything but the "base sample"
            scroll_speed = 0.0
        '''
        vert = """<div id="bigvertreport%s" class="container">
                      <div id="biganimatedvertreport%s" class= "vertical" data-charttime=%s></div>\n</div>\n"""
        button_html = vert % (str(report_index_value), str(report_index_value), end_time)

        '''
        if report_index_value == 0:
        else:
            button_html = ""
        '''

        info_tag = '<div class="section" id="biginforeport%s">\n' % str(report_index_value)
        if match_info[1] == "":
            info_tag_core = "\nSource file: {}<br>\nThis is the data we matched against!<br>\n".format(match_info[0].split(os.sep)[-1])
        else:
            info_tag_core = "\nMatched against file: {}<br>\nMatch sequence: {}<br>\n".format(match_info[0].split(os.sep)[-1], match_info[1])

        info_tag += info_tag_core
        info_tag += '\n</div>\n'
        # if we reverse the list, we reverse the axis
        report_string = t.substitute(PAGE_NAME=page_name, JAVASCRIPT_NOTE_DATA=javascript_note_data_string, LANE_NAMES=str([LANES_LOOKUP[i] for i in range(N_LANES)]), LANE_TIME_END=end_time, INFO_TAG=info_tag, BUTTON_HTML=button_html, BUTTON_FUNCTION=button_function, REPORT_NAME="report{}".format(report_index_value))
        final_report_string += report_string

    invis_selection_dropdown_string_pre = """
	<div id="dropdown_invis" display="none">
        <select id="reportBox%s" onchange="reportChangeFunc%s();" display="none">
            <option value="Select a preset report" display="none">Select a report</option>
    """

    selection_dropdown_string_pre = """
	<div id="dropdown" display="inline-block">
        <select id="reportBox%s" onchange="reportChangeFunc%s();">
            <option value="Select a preset report">Select a report</option>
    """

    selection_dropdown_string_post = """
        </select>
	</div>
    """

    selection_dropdown_string1 = selection_dropdown_string_pre % (str(1), str(1)) + selection_dropdown_string_core + selection_dropdown_string_post
    selection_dropdown_string2 = selection_dropdown_string_pre % (str(2), str(2)) + selection_dropdown_string_core + selection_dropdown_string_post
    selection_dropdown_string3 = selection_dropdown_string_pre % (str(3), str(3)) + selection_dropdown_string_core + selection_dropdown_string_post
    shadow_selection_dropdown_string = invis_selection_dropdown_string_pre % ("Shadow", "Shadow") + selection_dropdown_string_core + selection_dropdown_string_post + "\n"

    selection_dropdown_string = shadow_selection_dropdown_string
    selection_dropdown_string += '\n<table>\n    <tr>\n'
    selection_dropdown_string += '<td><button onclick="clearReportFunction()">Clear Reports</button></td>\n'
    selection_dropdown_string += '    <td>\n' + selection_dropdown_string1 + "\n    </td>\n"
    selection_dropdown_string += '    <td>\n' + selection_dropdown_string2 + "\n    </td>\n"
    selection_dropdown_string += '    <td>\n' + selection_dropdown_string3 + "\n    </td>\n"
    selection_dropdown_string += '    </tr>\n' + '</table>\n'
    return midijs_str + webaudio_str + midi_player_string + selection_dropdown_string + final_report_string


def hodl():
    from string import Template
    plot_module_path = __file__
    plot_module_dir = str(os.sep).join(os.path.split(plot_module_path)[:-1])
    with open(plot_module_dir + os.sep + "index_template.html", "r") as f:
        l = f.read()

    report_first_div_template = '\n<div class="first-container" id="big{}_index" display="none"></div>\n'
    report_secondary_div_template = '\n<div class="secondary-container" id="big{}_index" display="none"></div>\n'
    report_load_template = "<script>\n$('#big{}_index').load('{}.html');"
    report_load_template += '$("#big{}_index").hide();\n</script>\n'
    report_button_function_template = """
    <script>
    function toggleBig%sIndexFunction() {
        var btn = document.getElementById("toggleBig%sIndex")
        var x = document.getElementById("big%s_index");
        btn.style.borderStyle = (btn.style.borderStyle!=='inset' ? 'inset' : 'outset')
        if (x.style.display === "none") {
            x.style.display = "block";
        } else {
            x.style.display = "none";
        }
    }
    </script>
    """
    report_button_html_template = '<button id="toggleBig%sIndex" onclick="toggleBig%sIndexFunction()">%s (%s)</button>'
    index_chunk = "\n"
    # do buttons, then divsm then functions
    for tup in list_of_report_file_base_name_tuples:
        name = tup[0]
        data_fname = tup[1]
        index_chunk = index_chunk + report_button_function_template % (name, name, name)
        index_chunk = index_chunk + report_button_html_template % (name, name, name, data_fname)

    for _i, tup in enumerate(list_of_report_file_base_name_tuples):
        name = tup[0]
        if _i == 0:
            index_chunk = index_chunk + report_first_div_template.format(name)
        else:
            index_chunk = index_chunk + report_secondary_div_template.format(name)

    for tup in list_of_report_file_base_name_tuples:
        name = tup[0]
        index_chunk = index_chunk + report_load_template.format(name, name, name)
        index_chunk = index_chunk + "\n"
    '''
    report_div_template = '<div id="{}"></div>\n'
    report_load_template = "    $('#{}').load('{}.html');\n"
    index_chunk = "\n"
    # do divs
    for name in list_of_report_file_base_names:
        index_chunk = index_chunk + report_div_template.format(name)
    # do script tag
    index_chunk = index_chunk + "<script>\n"
    # do loads
    for name in list_of_report_file_base_names:
        index_chunk = index_chunk + report_load_template.format(name, name)
    # end script
    index_chunk = index_chunk + "</script>\n"
    '''

    # unused example of what it should look like
    """
    example_index_chunk = '''
    <div id="report1"></div>
    <div id="report2"></div>

    <script>
        $('#report1').load('report1.html');
        $('#report2').load('report2.html');
    </script>
    '''
    """
    t = Template(l)
    return t.substitute(INDEX_BODY=index_chunk)


def make_index_html_string(list_of_report_file_base_name_tuples):
    from string import Template
    plot_module_path = __file__
    plot_module_dir = str(os.sep).join(os.path.split(plot_module_path)[:-1])
    with open(plot_module_dir + os.sep + "index_template.html", "r") as f:
        l = f.read()

    report_first_div_template = '\n<div class="first-container" id="big{}_index" display="none"></div>\n'
    report_secondary_div_template = '\n<div class="secondary-container" id="big{}_index" display="none"></div>\n'
    report_load_template = "<script>\n$('#big{}_index').load('{}.html');"
    report_load_template += '$("#big{}_index").hide();\n</script>\n'
    report_button_function_template = """
    <script>
    function toggleBig%sIndexFunction() {
        var btn = document.getElementById("toggleBig%sIndex")
        var x = document.getElementById("big%s_index");
        btn.style.borderStyle = (btn.style.borderStyle!=='inset' ? 'inset' : 'outset')
        if (x.style.display === "none") {
            x.style.display = "block";
        } else {
            x.style.display = "none";
        }
    }
    </script>
    """
    report_button_html_template = '<button id="toggleBig%sIndex" onclick="toggleBig%sIndexFunction()">%s (%s)</button>'
    index_chunk = "\n"
    # do buttons, then divsm then functions
    for tup in list_of_report_file_base_name_tuples:
        name = tup[0]
        data_fname = tup[1]
        index_chunk = index_chunk + report_button_function_template % (name, name, name)
        index_chunk = index_chunk + report_button_html_template % (name, name, name, data_fname)

    for _i, tup in enumerate(list_of_report_file_base_name_tuples):
        name = tup[0]
        if _i == 0:
            index_chunk = index_chunk + report_first_div_template.format(name)
        else:
            index_chunk = index_chunk + report_secondary_div_template.format(name)

    for tup in list_of_report_file_base_name_tuples:
        name = tup[0]
        index_chunk = index_chunk + report_load_template.format(name, name, name)
        index_chunk = index_chunk + "\n"
    '''
    report_div_template = '<div id="{}"></div>\n'
    report_load_template = "    $('#{}').load('{}.html');\n"
    index_chunk = "\n"
    # do divs
    for name in list_of_report_file_base_names:
        index_chunk = index_chunk + report_div_template.format(name)
    # do script tag
    index_chunk = index_chunk + "<script>\n"
    # do loads
    for name in list_of_report_file_base_names:
        index_chunk = index_chunk + report_load_template.format(name, name)
    # end script
    index_chunk = index_chunk + "</script>\n"
    '''

    # unused example of what it should look like
    """
    example_index_chunk = '''
    <div id="report1"></div>
    <div id="report2"></div>

    <script>
        $('#report1').load('report1.html');
        $('#report2').load('report2.html');
    </script>
    '''
    """
    t = Template(l)
    return t.substitute(INDEX_BODY=index_chunk)


def test_example():
    qppm = 220
    #qpps = qppm / 60.
    quarter_time_const_s = .5
    # notes should be note value, start time, duration
    # we add the name internally to be consistent
    import json
    with open("bwv101.7.C-minor-transposed.json", "r") as f:
        music_json_data = json.load(f)

    l = []
    for _p in range(len(music_json_data["parts"])):
        parts = music_json_data["parts"][_p]
        parts_times = music_json_data["parts_times"][_p]
        parts_cumulative_times = music_json_data["parts_cumulative_times"][_p]
        assert len(parts) == len(parts_times)
        assert len(parts_times) == len(parts_cumulative_times)
        for _s in range(len(parts)):
            d = parts_times[_s]
            l.append((parts[_s], parts_cumulative_times[_s] - d, d))

    last_step = max([t[1] for t in l])
    last_step_dur = max([t[2] for t in l if t[1] == last_step])
    end_time = last_step + last_step_dur

    r = make_plot_json(l)
    """
    # write out the json + javascript
    with open("notesOnlyJSON.js", "w") as f:
        f.write(r)
    """

    report_dir = "test_report"
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    # write out the index.html with minor modifications for lane names
    w = make_website_string(javascript_note_data_string=r, end_time=end_time)
    with open(report_dir + os.sep + "report1.html", "w") as f:
        f.write(w)

    # test example, every note 
    l = []
    start = 0.
    for i in range(20, 108):
        l.append((i, start, 1.))
        start += 1.

    last_step = max([t[1] for t in l])
    last_step_dur = max([t[2] for t in l if t[1] == last_step])
    end_time = last_step + last_step_dur
    r = make_plot_json(l)

    w = make_website_string(javascript_note_data_string=r, end_time=end_time)
    with open(report_dir + os.sep + "report2.html", "w") as f:
        f.write(w)

    # no report dir because the load is relative to the index file we write out!
    w = make_index_html_string(["report1", "report2"])
    #w = make_index_html_string(["report2"])
    with open(report_dir + os.sep + "index.html", "w") as f:
        f.write(w)


if __name__ == "__main__":
    test_example()
