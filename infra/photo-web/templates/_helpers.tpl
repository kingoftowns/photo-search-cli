{{- define "photo-web.fullname" -}}
photo-web
{{- end }}

{{- define "photo-web.labels" -}}
app.kubernetes.io/name: {{ include "photo-web.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "photo-web.selectorLabels" -}}
app.kubernetes.io/name: {{ include "photo-web.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
