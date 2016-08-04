$(document).ready(function() {
    $("form").on('submit', function() {
        $.ajax({url: "/model",
                data: {'product': $("input[name='product']").val(),
                        'component': $("input[name='component']").val(),
                        'assignee': $("input[name='assignee']").val(),
                        'cc': $("input[name='cc']").val(),
                        'short_desc': $("input[name='short_desc']").val(),
                        'op_sys': $("input[name='op_sys']").val(),
                        'desc': $("textarea[name='desc']").val()
                },
                async: false,
                success: function(result) {
                    $("span#model_result").html(result)
                }})
    })
});
