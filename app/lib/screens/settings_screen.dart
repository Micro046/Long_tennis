import 'package:flutter/material.dart';

import '../api/client.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _url = TextEditingController(text: ApiClient.instance.baseUrl);
  final _token = TextEditingController(text: ApiClient.instance.token);
  String? _status;
  bool _busy = false;

  Future<void> _test() async {
    setState(() => _busy = true);
    await ApiClient.instance.configure(_url.text, _token.text);
    try {
      final h = await ApiClient.instance.health();
      await ApiClient.instance.sessions(); // checks the token too
      _status = 'Connected: server ${h['version']} on ${h['device']}, ${h['queue']} job(s) queued.';
    } catch (e) {
      _status = describeError(e);
    }
    setState(() => _busy = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Server')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        const Text('Analysis runs on your server (the Mac for testing, later the GPU laptop). '
            'Enter its address on the local network, for example http://192.168.1.20:8770'),
        const SizedBox(height: 16),
        TextField(controller: _url, decoration: const InputDecoration(labelText: 'Server address', border: OutlineInputBorder()),
            keyboardType: TextInputType.url),
        const SizedBox(height: 12),
        TextField(controller: _token, decoration: const InputDecoration(labelText: 'Access token', border: OutlineInputBorder())),
        const SizedBox(height: 16),
        FilledButton.icon(onPressed: _busy ? null : _test, icon: const Icon(Icons.wifi_tethering), label: const Text('Save and test')),
        if (_status != null) Padding(padding: const EdgeInsets.only(top: 16), child: Text(_status!)),
      ]),
    );
  }
}
